You can use the transformers package to use Snowflake's arctic-embed model, as shown below. For optimal retrieval quality, use the CLS token to embed each text portion and use the query prefix below (just on the query).

import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
model.eval()

query_prefix = 'query: '
queries  = ['what is snowflake?', 'Where can I get the best tacos?']
queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=8192)

documents = ['The Data Cloud!', 'Mexico City of Course!']
document_tokens =  tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=8192)

# Compute token embeddings
with torch.no_grad():
    query_embeddings = model(**query_tokens)[0][:, 0]
    document_embeddings = model(**document_tokens)[0][:, 0]


# normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    #Output passages & scores
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)

This should produce the following scores

Query: what is snowflake?
tensor(0.2715) The Data Cloud!
tensor(0.0661) Mexico City of Course!
Query: Where can I get the best tacos?
tensor(0.2797) Mexico City of Course!
tensor(0.1250) The Data Cloud!

Using Huggingface Transformers.js
If you haven't already, you can install the Transformers.js JavaScript library from NPM using:

npm i @huggingface/transformers

You can then use the model for retrieval, as follows:

import { pipeline, dot } from '@huggingface/transformers';

// Create feature extraction pipeline
const extractor = await pipeline('feature-extraction', 'Snowflake/snowflake-arctic-embed-m-v2.0', {
    dtype: 'q8',
});

// Generate sentence embeddings
const sentences = [
    'query: what is snowflake?',
    'The Data Cloud!',
    'Mexico City of Course!',
]
const output = await extractor(sentences, { normalize: true, pooling: 'cls' });

// Compute similarity scores
const [source_embeddings, ...document_embeddings ] = output.tolist();
const similarities = document_embeddings.map(x => dot(source_embeddings, x));
console.log(similarities); // [0.24783534471401417, 0.05313122704326892]

