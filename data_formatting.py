from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


def get_tokenizer_and_data_collator_and_prompt_formatting(model_name: str, tokenizer_name: str):
    
    def formatting_prompts_func(example):
        #TODO Use the same prompt for PISTOL changing key values for the JSON
        output_texts = []
        # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
        for i in range(len(example['instruction'])):
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {example['instruction'][i]}
            
            ### Input:
            {example['input'][i]}
            
            ### Response:
            {example['output'][i]}
            '''
            output_texts.append(text)
        return output_texts

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True,trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "### Response:" # alpaca response tag
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    return tokenizer, data_collator, formatting_prompts_func
