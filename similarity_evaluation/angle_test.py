from angle_emb import AnglE, Prompts

# init
angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')

# set prompt
print('All predefined prompts:', Prompts.list_prompts())
angle.set_prompt(prompt=Prompts.A)
print('prompt:', angle.prompt)

# encode text
vec = angle.encode({'text': 'hello world'}, to_numpy=True)
print(vec)
vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
print(vecs)
