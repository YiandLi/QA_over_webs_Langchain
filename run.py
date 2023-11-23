from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # Load data
    loader = AsyncChromiumLoader(["https://www.trustradius.com/products/thoughtspot/competitors#small-businesses",
                                  "https://www.trustradius.com/products/thoughtspot/competitors"])
    
    html = loader.load()
    contents = []  # {}
    contents_hash_set = set()
    
    # pre process --  better for load into relational Database
    for page in html:
        soup = BeautifulSoup(page.page_content, 'html.parser')
        page_label = " ".join(
            [i.text for i in soup.find('h1', class_='ProductHeader_product-name__yY2gU').find_all("span")])
        
        target_div = soup.find('div', class_='ProductAlternativesContent_alternatives-layout__u6hEm')
        section_elements = target_div.find_all('section')
        section_elements.pop(0)
        
        for i in section_elements:  # loop through Best ThoughtSpot types
            cmps = i.find('div').find_all('article')
            heads = [i.find('h3').text for i in cmps]
            des = [i.find('p').text for i in cmps]
            award = i.find('header').text
            features = [ii.text for ii in i.find('div', class_='Features_list__JpwtQ').find_all("li")]
            # contents[award] = [f"company name: {i}\nAward title: {award}\ndescription: {j}" for i, j in zip(heads, des)]
            
            doc_content = [f"company name: {i}\n" \
                           f"{'description: ' + j if j else ''}\n" \
                           f"Feature: {' '.join(features)}" for i, j in zip(heads, des)]
            
            for doc in doc_content:
                if hash(doc) not in contents_hash_set:
                    contents_hash_set.add(hash(doc))
                    contents.append(
                        Document(
                            page_content=doc,
                            metadata={'url': page.metadata, 'highlight label': page_label, 'Award title': award})
                    )
            
            contents_hash_set.add(hash(doc))
            contents.append(
                Document(
                    page_content=f"{page_label}, List of companies with Award title '{award}': {', '.join(heads)}",
                    metadata={'url': page.metadata})
            )
            del doc_content, heads, award, des, features, cmps
    
    # get NoSQL vector db
    embedding_model_name = 'BAAI/bge-base-en-v1.5'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_documents(contents, embeddings)
    
    prompt_template = """基于以下已知信息，简洁，专业和细致地回答用户的问题。已知内容 是关于各种企业软件、产品和服务的真实用户评价和经验。
                         要求：
                            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。
                            不允许在答案中添加编造成分。答案请使用中文。
                            请一次性返回合并所有的获选结果，即结果使用1，2，3序号进行最终合并 。
                         已知内容: {context}
                         问题: {question} """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    # get chain
    chain = RetrievalQA.from_llm(
        llm=ChatOpenAI(temperature=0),
        prompt=prompt,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5})  # k=3 for less cost
    )
    chain.return_source_documents = True
    
    # test
    for query in [
        '所有的Best ThoughtSpot Alternatives for Small Businesses有哪些？',
        '所有的ThoughtSpot Competitors and Alternatives有哪些？',
        'GoodData的云端数据和分析平台具有哪些关键特性和优势？它如何利用自动化和人工智能？',
        'Infor Birst的关键特性是什么？它支持在哪些部署环境中使用？'
    ]:
        response = chain(query)
        print(f"Query:\n\t{query}\n\n"
              f"Response:\n\t{response['result']}\n\n"
              f"Source:\n\t{response['source_documents']}")
        print("\n")

"""


"""
