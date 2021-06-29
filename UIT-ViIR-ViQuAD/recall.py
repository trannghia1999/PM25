




import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-11.0.11"
from pygaggle.rerank.base import Text,Query
from pygaggle.rerank.transformer import MonoBERT

reranker =  MonoBERT()
import pandas as pd
text_file = pd.read_json ('texts.json',typ='series')
question_file = pd.read_json ('questions.json',typ='series')
result_file = pd.read_json ('retrieved_results.json',typ='series')

text_df = pd.DataFrame({'document_id':text_file.index, 'document':text_file.values})
question_df = pd.DataFrame({'question_id':question_file.index, 'question':question_file.values})
result_df = pd.DataFrame({'question_id':result_file.index, 'document_id':result_file.values})


passages = []
for i in range(text_df.shape[0]):
    x=[]
    x.append(text_df['document_id'][i])
    x.append(text_df['document'][i])
    passages.append(x)

texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]

# for i in range(0, 10):
#     print(f'{i+1:2} {texts[i].metadata["docid"]:15} {texts[i].score:.5f} {texts[i].text}')

p=231
query = Query(question_df['question'][p])
reranked = reranker.rerank(query, texts)

# Print out reranked results:
print(question_df['question_id'][p])
print(question_df['question'][p])

# for i in range(0, 10):
#     print(f'{i+1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f} {reranked[i].text}')

import time
start_time = time.clock()

queries=[]
documents=[]
scores = []
d=[]
#for i in range(question_df.shape[0]):
for i in range(100):
    id = question_df['question_id'][i]
    query = Query(question_df['question'][i])
    reranked = reranker.rerank(query, texts)
    for j in range(0, 10):
        queries.append(id)
        documents.append(reranked[j].metadata["docid"])
        scores.append(reranked[j].score)
        d.append(reranked[j].text)

# print(time.clock() - start_time, "seconds")

query_output = pd.DataFrame({'question_id':queries,'document_id':documents,'score':scores})
query_output["rank"] = query_output.groupby("question_id")["score"].rank(ascending=0,method='dense')
query_output["rank"] = query_output['rank'].astype(int)

output = query_output[['question_id','document_id','rank']]
output.columns = ['query','document','rank']
