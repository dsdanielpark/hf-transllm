import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from translang import TranslationService


class LLMtranslator(TranslationService):
    """
    LLMtranslator is a translation service based on the Llama language model.
    It extends the TranslationService class and provides translation functionality using Llama.

    Args:
        model_path (str): The path to the Llama model.
        target_lang (str, optional): The target language for translation (default: "en").
        translator (str, optional): The translation service to use (default: "google").
        torch_dtype (torch.dtype, optional): The data type for torch (default: torch.float16).
        device_map (str, optional): The device map for torch (default: "auto").
        deepl_api_key (str, optional): DeepL API key (if required).
        bard_api_key (str, optional): Bard API key (if required).
        openai_model (str, optional): OpenAI model name (default: "gpt-3.5-turbo").
        openai_api_key (str, optional): OpenAI API key (if required).
    """

    def __init__(
        self,
        model_path,
        target_lang="en",
        translator="google",
        torch_dtype=torch.float16,
        offload_folder=None,
        device_map="auto",
        google_api_key=None,
        deepl_api_key=None,
        bard_api_key=None,
        openai_api_key=None,
        openai_model="gpt-3.5-turbo",
    ):
        super().__init__(
            translator=translator,
            google_api_key=google_api_key,
            deepl_api_key=deepl_api_key,
            bard_api_key=bard_api_key,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
        )
        self.model_path = model_path
        self.target_lang = target_lang
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            offload_folder=offload_folder,
        )
        self.model.tie_weights()

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

        # Different from this class's generate method is the generate method of the Hugging Face model.
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
        if self.target_lang == "en":
            return self.inference(prompt)
        else:
            translated_prompt = self.process_prompt(prompt)
            answer = self.inference(translated_prompt)
            translated_answer = self.process_answer(answer)
        return translated_answer
