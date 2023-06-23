import torch
import deepl
import openai
from transformers import LlamaTokenizer, LlamaForCausalLM
from googletrans import Translator
from bardapi import Bard
import re

class LLMtranslator:
    """
    LLMtranslator is a translation and generation model that translates sentences into different languages and generates text based on the translated input.

    Args:
        model_path (str): The path or name of the pre-trained LLM model.
        target_lang (str, optional): The target language for translation. Defaults to 'ko'.
        translator (str, optional): The translation service to use. Supported options are 'google', 'deepl', and 'bard'. Defaults to 'google'.
        torch_dtype (torch.dtype, optional): The torch data type for model inference. Defaults to torch.float16.
        device_map (str, optional): The device mapping for model inference. Defaults to 'auto'.
        deepl_api (str, optional): The API key for DeepL translation service. Required if translator is set to 'deepl'.
        bard_api (str, optional): The API key for Bard translation service. Required if translator is set to 'bard'.
        openai_api (str, optional): The API key for OpenAI translation service. Required if translator is set to 'openai'.
        openai_model (str, optional): The model for OpenAI translation service. Required if translator is set to 'openai'.

    Methods:
        translate_text(text, dest_lang):
            Translates the given text to the specified destination language.

        translate_to_en(text):
            Translates the given text to English.

        translate_to_targetlang(text):
            Translates the given text to the target language.

        generate(prompt):
            Generates text based on the provided prompt and returns it in the target language.
    """

    def __init__(self, model_path, target_lang='ko', translator='google', torch_dtype=torch.float16, device_map='auto', deepl_api=None, bard_api=None, openai_api=None, openai_model='gpt-3.5-turbo'):
        self.model_path = model_path
        self.target_lang = target_lang
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device_map)
        self.translator = translator
        self.bard_api = bard_api
        self.openai_api = openai_api
        self.deepl_api = deepl_api
        self.openai_model = openai_model
        if self.translator == 'google':
            self.translator_obj = Translator()
        elif self.translator == 'deepl':
            self.translator_obj = deepl.Translator(self.deepl_api)
        elif self.translator == 'bard':
            self.translator_obj = Bard(token=self.bard_api)
            

    def translate(self, text: str, dest_lang: str) -> str:
        """
        Translates the given text to the specified destination language.

        Args:
            text (str): The text to be translated.
            dest_lang (str): The destination language code.

        Returns:
            str: The translated text in the specified language.
        """
        if self.translator == 'google':
            return self.translator_obj.translate(text, dest=dest_lang)
        elif self.translator == 'deepl':
            return self.translator_obj.translate_text(text, target_lang=dest_lang)
        elif self.translator == 'bard':
            translated = self.translator_obj.get_answer(f'{text}를 {dest_lang}로 번역해.')
            extracted_text = re.findall(r'"([^"]*)"', translated)
            return extracted_text[0]
        elif self.translator == 'openai':
            response = openai.ChatCompletion.create(
                                                    model=self.openai_model,
                                                    messages=[{"role": "user", "content": f'{text}를 {dest_lang}으로 번역해'}]
                                                    )
            return response

    def translate_to_en(self, text: str) -> str:
        """
        Translates the given text to English.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text in English.
        """
        return self.translate(text, 'en')

    def translate_to_targetlang(self, text: str) -> str:
        """
        Translates the given text to the target language.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text in the target language.
        """
        return self.translate(text, self.target_lang)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt and returns it in the target language.

        Args:
            prompt (str): The prompt for text generation.

        Returns:
            str: The generated text in the target language.
        """
        translated_prompt = self.translate_to_en(prompt)
        input_ids = self.tokenizer(translated_prompt, return_tensors="pt").input_ids
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=32)
        answer = self.tokenizer.decode(generation_output[0])
        translated_answer = self.translate_to_targetlang(answer)
        return translated_answer
