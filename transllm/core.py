import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from translang import TranslationService


class LLMtranslator(TranslationService):
    """
    LLMtranslator is a translation service based on the Llama language model.
    It extends the TranslationService class and provides translation functionality using Llama.

    Args:
        model_path (str): The path to the Llama model.
        target_lang (str, optional): The target language for translation (default: "ko").
        translator (str, optional): The translation service to use (default: "google").
        torch_dtype (torch.dtype, optional): The data type for torch (default: torch.float16).
        device_map (str, optional): The device map for torch (default: "auto").
        deepl_api (str, optional): DeepL API key (if required).
        bard_api (str, optional): Bard API key (if required).
        openai_model (str, optional): OpenAI model name (default: "gpt-3.5-turbo").
        openai_api (str, optional): OpenAI API key (if required).
    """

    def __init__(
        self,
        model_path,
        target_lang="ko",
        translator="google",
        torch_dtype=torch.float16,
        device_map="auto",
        deepl_api=None,
        bard_api=None,
        openai_model="gpt-3.5-turbo",
        openai_api=None
    ):
        super().__init__(
            translator=translator,
            deepl_api=deepl_api,
            bard_api=bard_api,
            openai_api=openai_api,
            openai_model=openai_model,
        )
        self.model_path = model_path
        self.target_lang = target_lang
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map
        )

    def process_prompt(self, prompt: str) -> str:
        """
        Preprocesses the prompt text for translation.

        Args:
            prompt (str): The prompt text.

        Returns:
            str: The preprocessed prompt text.
        """
        return self.translate(prompt, "en")

    def process_answer(self, answer: str) -> str:
        """
        Postprocesses the translated answer.

        Args:
            answer (str): The translated answer.

        Returns:
            str: The postprocessed answer.
        """
        return self.translate(answer, self.target_lang)

    def inference(self, prompt: str) -> str:
        """
        Generates the translated answer based on the given prompt.

        Args:
            prompt (str): The prompt text.

        Returns:
            str: The translated answer.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            generation_output = self.model.generate(input_ids=input_ids, max_length=32)
        answer = self.tokenizer.decode(generation_output[0])
        return answer

    def generate(self, prompt: str) -> str:
        """
        Generates the translated answer for the given prompt.

        Args:
            prompt (str): The prompt text.

        Returns:
            str: The translated answer.
        """
        translated_prompt = self.process_prompt(prompt)
        answer = self.inference(translated_prompt)
        translated_answer = self.process_answer(answer)
        return translated_answer
