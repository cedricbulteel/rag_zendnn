from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import time
from prompts import questions

import random
import pandas as pd



def benchmark(batch_size : int, backend : str):

    # default
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                            model_kwargs=model_kwargs,
                                            encode_kwargs=encode_kwargs,
                                            compile_backend=backend) #custom arg

    collection_name = "collection_of_webpages"

    vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_model, persist_directory="./vectordb")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    print(f"Vectorstore {collection_name} ready!")

    def format_docs(docs_batch):

        return ["\n\n".join(doc.page_content for doc in docs) for docs in docs_batch]

    # fill in your LLM server URL, vLLM example: 'http://localhost:8000/v1'
    llm = ChatOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    openai_api_key="EMPTY",
                    base_url="<your url>",
                    verbose=True
                    )

    template = """Answer the question based only on the following context and always end the answer with 'Great question!':
    {context}
    - -
    Answer the question based on the above context: {input_prompt}
    """

    input_prompts = random.sample(questions, batch_size)
    print(f'Randomly selected {len(input_prompts)} input prompts')

    t_retr_s = time.time()
    docs_batch = retriever.batch(input_prompts)
    t_retr_e = time.time()

    contexts = format_docs(docs_batch)

    filled_prompts = [template.format(context=c, input_prompt=p) for c,p in zip(contexts, input_prompts)]


    t_gen_s = time.time()
    response = llm.batch(filled_prompts)
    t_gen_e = time.time()

    total_tokens = sum([message.response_metadata['token_usage']['total_tokens'] for message in response])

    t_gen = t_gen_e - t_gen_s
    t_retr = t_retr_e - t_retr_s
    t_tot = t_gen + t_retr

    token_throughput = total_tokens/t_gen
    t_per_token = t_gen/total_tokens * 10**3 #ms/token

    output_str = f'''
    Results for batch size {batch_size} and backend {backend}: \n

        Total retrieval time: {t_retr} s
        Total LLM time (generation + prefill): {t_gen} s 
        Total time: {t_tot}\n

        Token throughput: {token_throughput} tokens/s
        Time per token: {t_per_token} ms/token

    '''

    print(output_str)

    return [backend, batch_size, t_retr, t_gen, t_tot, token_throughput, t_per_token]


batch_sizes = [1,2,4,8,16,32,64,128]
backends = ['torch', 'zentorch']

if __name__ == "__main__":

    for i in range(10):

        print(f"iteration {i+1}")
    
        results = []
        for b in backends:
            for bs in batch_sizes:
                print("#"*10)
                print(f'Benchmarking with backend {b} and batch size {bs}')
                print("#"*10)

                try:
                    res = benchmark(batch_size=bs, backend=b)

                    results.append(res)
                except BaseException as e:
                    print(f'An error occured: {e}')
                    results.append([0]*7)


        df = pd.DataFrame(data=results,
            columns=['backend', 'batch_size', 't_retr (s)', 't_gen (s)', 't_tot (s)', 'throughput (t/s)', 'time_per_token (ms/t)']
            )
        
        print(df)
        
        df.to_csv(f'rag_benchmark_{i+1}.csv')

    