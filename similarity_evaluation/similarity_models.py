import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from angle_emb import AnglE, Prompts, AngleDataTokenizer
from angle_emb.utils import cosine_similarity
from transformers import AutoTokenizer
from datasets import Dataset



# define a similarity model class
class SimilarityModel:
    def __init__(self, name):
        self.name = name
    
    def compute_similarity(self, text1, text2):
        # Placeholder for actual similarity computation logic
        pass
    

class SpacySimilarityModel(SimilarityModel):
    def __init__(self):
        super().__init__('spacy')
        self.nlp = spacy.load("en_core_web_lg")

    def compute_similarity(self, text1, text2):
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

transformer_models = [
    'sentence-transformers/all-MiniLM-L12-v2',
    "cross-encoder/stsb-distilroberta-base",
    "neuml/pubmedbert-base-embeddings",
    # too big for gpu
    # "epfl-llm/meditron-7b",
    'distilbert-base-nli-mean-tokens',
    'FremyCompany/BioLORD-2023'
]

# Sentence Transformer similarity model
class SentenceTransformerSimilarityModel(SimilarityModel):
    def __init__(self, model_name):
        super().__init__(f'sentence_transformer_{model_name}')
        self.model = SentenceTransformer(model_name)

    # TODO: could also be fine-tuned: https://github.com/adiekaye/fine-tuning-sentence-transformers/blob/main/01_tuning_your_model.py 

    def compute_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

encoder_models = [
    'sentence-transformers/all-MiniLM-L12-v2',
    "cross-encoder/stsb-distilroberta-base",
    "neuml/pubmedbert-base-embeddings",
    # too big for gpu
    # "epfl-llm/meditron-7b",
    'FremyCompany/BioLORD-2023'
]

# Cross Encoder similarity model
class CrossEncoderSimilarityModel(SimilarityModel):
    def __init__(self, model_name):
        super().__init__(f'cross_encoder_{model_name}')
        self.model = CrossEncoder(model_name)
    
    def compute_similarity(self, text1, text2):
        return float(self.model.predict([[text1, text2]])[0])
    
# angLE model
class AngLEModel(SimilarityModel):
    def __init__(self):
        super().__init__('angle')
        self.model = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                              pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                              pooling_strategy='last',
                              is_llm=True,
                              torch_dtype=torch.float16).cuda()
        
    def train(self, train_df, val_df):
        # data columns should be mapped from 'sentence1', 'sentence2', 'score' to text1, text2, and label
        train_data = train_df.rename(columns={
            'sentence1': 'text1',
            'sentence2': 'text2',
            'score': 'label'
        })
        val_data = val_df.rename(columns={
            'sentence1': 'text1',
            'sentence2': 'text2',
            'score': 'label'
        })
        # set as hf dataset
        train_dataset = Dataset.from_pandas(train_data).shuffle().map(AngleDataTokenizer(self.model.tokenizer, self.model.max_length), num_proc=8)
        val_dataset = Dataset.from_pandas(val_data).map(AngleDataTokenizer(self.model.tokenizer, self.model.max_length), num_proc=8)

        self.model.fit(
            train_ds=train_dataset,
            valid_ds=val_dataset,
            output_dir='ckpts/sts-b',
            batch_size=32,
            epochs=5,
            learning_rate=2e-5,
            save_steps=100,
            eval_steps=1000,
            warmup_steps=0,
            gradient_accumulation_steps=1,
            loss_kwargs={
                'cosine_w': 1.0,
                'ibn_w': 1.0,
                'cln_w': 1.0,
                'angle_w': 0.02,
                'cosine_tau': 20,
                'ibn_tau': 20,
                'angle_tau': 20
            },
            fp16=True,
            logging_steps=100
        )


    def compute_similarity(self, text1, text2):
        vec1, vec2 = self.model.encode([
            {'text': text1},
            {'text': text2}
        ], prompt=Prompts.A)
        return cosine_similarity(vec1, vec2)