#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASENAME = Path(__file__).stem
SYSTEM_PROMPT = """\
- Role: Expert in conversation understanding and information retrieval.
- Background: When users retrieve information, they often omit or use pronouns in the conversation, which may lead to incomplete or vague query sentences. In order to improve the accuracy and efficiency of retrieval, it is necessary to complete the user's query statement, make it independent of the session context, and correctly handle the topic conversion.
- Profile: You are an expert proficient in natural language processing and information retrieval. You can deeply analyze the dialog context, understand the user's real intention, and convert fuzzy query sentences into accurate and context independent queries.
- Skills: You have the key abilities of conversation analysis, semantic understanding, anaphora resolution, topic tracking and information retrieval. You can accurately complete the omitted parts, handle the pronouns, and maintain the consistency and accuracy of the query during topic conversion.
- Goals: According to the historical dialog context, the current user question and the system reply, complete the omitted and referred parts of the user query sentence, correctly handle the topic conversion phenomenon, and finally output an independent query with complete information and unrelated to the session context for information retrieval.
- Constrains: The complete query sentence must be clear and accurate to avoid ambiguity; When dealing with topic conversion, it is necessary to ensure that the query statement is consistent with the user's real intention; The output query statement shall be concise and clear, which is convenient for the information retrieval system to understand and execute.
- OutputFormat: Note that you should always try to rewrite it. Output a complete and standalone query statement, in the format of `Rewrite: $Rewrite`."""


class CustomLLMError(Exception):
    pass


class CustomOpenAI:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url
        self.client = self._init_client(api_key, base_url)

    @staticmethod
    def _init_client(api_key: Optional[str], base_url: Optional[str]):
        if not api_key or not base_url:
            raise CustomLLMError(f"[{BASENAME}] Please provide a valid API key and base_url.")
        logger.debug(f"[{BASENAME}] Initialize the OpenAI client, base_url={base_url}")
        return OpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_not_exception_type(CustomLLMError),
    )
    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Optional[dict]:
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        if api_key and base_url:
            client = self._init_client(api_key, base_url)
        else:
            client = self.client

        kwargs.pop("stream", False)
        params = {"model": model, "messages": messages, **kwargs, "stream": False}
        try:
            response = client.chat.completions.create(**params)
            return response.model_dump()
        except Exception as e:
            logger.error(f"[{BASENAME}] OpenAI API Call Chat Completion Failed, Got: {e}")
            raise e

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_not_exception_type(CustomLLMError),
    )
    def rewrite_utterance_multi(self, model: str, prompt: str, n: int) -> List[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        response = self.create_chat_completion(model, messages, **{"temperature": 0.7, "top_p": 0.8, "n": n})
        rewrite_utt_list = [choice["message"]["content"] for choice in response["choices"]]
        rewrite_utt_list = [
            rewrite_utt.split("Rewrite:")[-1].strip("$").strip()
            for rewrite_utt in rewrite_utt_list
            if rewrite_utt and "Rewrite:" in rewrite_utt
        ]
        if len(rewrite_utt_list) >= int(n * 0.7):
            return rewrite_utt_list
        else:
            raise ValueError(f"[{BASENAME}] Response content successfully retrieved, but some formatting is incorrect.")


def build_zero_shot_prompt(queries: List[str], answers: List[str], new_question: str, new_response: str = ""):
    context = list()
    for i, (q, a) in enumerate(zip(queries, answers)):
        context.append(f"Question{str(i + 1)}: {q}\nResponse{str(i + 1)}: {a}")
    context = "\n".join(context)

    part1 = f"### Context Begin ###\n{context}\n### Context End ###"
    part2 = f"Current Question: {new_question}"
    if new_response and new_response != "UNANSWERABLE":
        part2 += f"\nCurrent Response: {new_response}"
        tail_instruction = "Now, you should give me the rewriter of the **Current Question** under the **Context** and the **Current Response**. Do not add any explanation, the output format should always be `Rewrite: $Rewrite`."
    else:
        tail_instruction = "Now, you should give me the rewriter of the **Current Question** under the **Context**. Do not add any explanation, the output format should always be `Rewrite: $Rewrite`."

    prompt = "\n\n".join([part1, part2, tail_instruction])
    return prompt
