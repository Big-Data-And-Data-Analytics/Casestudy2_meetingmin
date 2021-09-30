import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# nlp = spacy.load('en')
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from neo4j import GraphDatabase
import itertools
#from entity import entity_greet


from datetime import datetime

d = datetime.today().strftime('%Y-%m-%d')
print(d)

#print(entity_greet)

d = {'meeting': [1,2,3,4], 'Completed': ['I have completed reddis assignment which has to be submitted day after tomorrow. Mark has acknowledged the completion status of the module.The codes are completed and ready for deployment for the data engineering.','I have completed my assignments. I have completed creating accounts in Jira tool for everyone. I have successfully completed the data modelling in Neo4J. I have already done my part in data modelling.','We have achieved the milestone on time. Every has completed their data modelling and user stories. We have completed all other modules. I have got a signoff from stakeholder regarding user stories. Professor was happy about the user story completion. ','The codes are debugged and ready for deployment. Every one is confident about their parts and completion status. some of the features on our casestudy are tokenisation parts of speech tagging and entity recognition are achieved.'],'Todo': ['Lets divide the work for information system module. I will do data modelling in mongodb. I will ingest data in mongodb. I will do data modelling in Neo4J. Lets do this by next week. Lets do this for a success. I need to do my assignment on Reddis','I will do the data modelling in Reddis. I will create a user story for the information system module. I will also create those user stories. How are we on the progress. I need to complete that ingestion by next week.','Lets start the documentation of user stories created. Let us divide the user stories. I will do MongoDb part. I will take up Neo4J. I will then take Reddis. I am confident that I will be able to do documentation.','Lets start with PPT and final documentation. Lets all gather for mock presentation. Code submission in git is pending.Do we have reasons for those items we missed this week in our todos.Lets all practice once before presentation.'],'Issue':['I see a blue screen which is an issue. I have to get signoff from stakeholders. Please help me out in this assignment.','I have personal health issues. Mark is unable to join as he has internet connection issues. I have logged an ticket on JIRA access issue.','I have issue with respect to second user story. I need help on user story five. I again see a blue screen which is pathetic. Please can any one show me how to do this logging issue in JIRA.','I hope all our production issues are listed and kept ready for this meet.Linda has addressed our issue in deployment and has given a word to us.'],'Irrelevant':['Lets catch up for lunch neary by restaurant. We deserve this holiday. Yayyy.Hello. Hi. Bye. Bye bye.','Hi. Hi. I am good. We will go out today for dinner. I heard nearby mall is good in asian cuisine.Bye. Catch you guys later.','Hi hello how are you. Hello Team hope everyone is doing good. I am alright.Bye.Ciao.','I have my birthday this week.Bye. Bte. Hi. Hi.']}


df = pd.DataFrame(data=d)

data = df[['meeting','Completed','Todo','Issue','Irrelevant']]
data = data[data.meeting.notnull()]

list_theme = ['Completed','Todo','Issue','Irrelevant']
execution_commands = []

data_base_connection = GraphDatabase.driver(uri="bolt://localhost:7687",auth=("neo4j","mom"))
session = data_base_connection.session()

for i, val in enumerate(list_theme):
    neo4j_create_statement = "Create(Task_type:Task{name:'"+str(val)+"'})"
    execution_commands.append(neo4j_create_statement)
for i in execution_commands:
    session.run(i)

#LDA
for i,val in enumerate(list_theme):
    #print(val)
    vectorizer = CountVectorizer(analyzer='word',       
                             min_df=1,                        # minimum required occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             max_features=5000,             # max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
                            )

    data_vectorized = vectorizer.fit_transform(data[val])

    lda_model = LatentDirichletAllocation(n_components=8, # Number of topics
                                      learning_method='online',
                                      random_state=0,       
                                      n_jobs = -1  # Use all available CPUs
                                     )
    lda_output = lda_model.fit_transform(data_vectorized)


    def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=4):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=4)        

# Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    l = []
    for i in df_topic_keywords:
        l.append(list(df_topic_keywords[i]))
    #print(l)
    l = list(itertools.chain.from_iterable(l))
    l = list(set(l))
    print(l)
    execution_commands = []
    for i, v in enumerate(l):
        neo4j_create_statement = "Create(Topic_discussed:topic{name:'"+str(v)+"'})"
        execution_commands.append(neo4j_create_statement)
    for i in execution_commands:
        session.run(i)
    print(val)        
    execution_commands = []
    for i in l:
        #print(i)
        neo4j_create_statement = "Match (a:Task),(b:topic) WHERE a.name = '"+str(val)+"' AND b.name='"+str(i)+"'CREATE (a)-[r:has]->(b)"
        execution_commands.append(neo4j_create_statement)
    for i in execution_commands:
        session.run(i)  
