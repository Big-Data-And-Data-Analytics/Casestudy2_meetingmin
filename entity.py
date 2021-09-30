from numpy import greater_equal
import spacy
import pandas as pd
nlp=spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# nlp = spacy.load('en')
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from neo4j import GraphDatabase
import itertools

""" nlp.pipe_names """
""" article_text= Hi hello how are you. I am good how are you .I have completed mongodb setup,I have to start the Neo4j data modelling.I have issue with my system I see blue screen.Lets catch up for weekend.Lets go to a new restaurant which is opened last week.See you then Bye

doc=nlp(article_text)
for ent in doc.ents:
  print(ent.text,ent.label_) """
# Getting the pipeline component
ner=nlp.get_pipe("ner")
#Train data

train_data = [
    ("hello", {"entities": [(0, 5, "greet")]}),
    ("okay bye guys", {"entities": [(5, 8, "greet")]}),
    ("bye guys", {"entities": [(0, 7, "greet")]}),
    ("Hi hello how are you", {"entities": [(0, 2, "greet")]}),
    ("Hi hello how are you", {"entities": [(3, 8, "greet")]}),
    ("hello how are you", {"entities": [(0, 5, "greet")]}),
    ("Hello Team hope everyone is doing good", {"entities": [(0, 5, "greet")]}),
    ("Hi Mark Good to have you back with us", {"entities": [(0, 2, "greet")]}),
    ("Hey Lucy how is you team out there", {"entities": [(0, 3, "greet")]}),
    ("Hi team how are we going today", {"entities": [(0, 2, "greet")]}),
    ("Hey Linda glad you joined us today for this meeting", {"entities": [(0, 3, "greet")]}),
    ("I'm fine", {"entities": [(0, 7, "greet")]}),


    ("We have achieved the milestone on time", {"entities": [(8, 16, "Completed")]}),
    ("I've completed task1 yesterday", {"entities": [(5, 20, "Completed")]}),
    ("I've completed task1 sucessfully", {"entities": [(5, 31, "Completed")]}),
    ("Speech to text is completed sucessfully without any delay", {"entities": [(0, 27, "Completed")]}),
    ("BERT is done now and it works smooth", {"entities": [(8, 12, "Completed")]}),
    ("The codes are debugged and ready for deployment", {"entities": [(27, 46, "Completed")]}),
    ("Slides for the presentation to stakeholder is done and ready for trial now", {"entities": [(46, 60, "Completed")]}),
    ("some of the features on our casestudy are tokenisation parts of speech tagging and entity recognition are achieved", {"entities": [(106, 113, "Completed")]}),
    ("We are ready with a prototype to the stakeholder", {"entities": [(7, 12, "Completed")]}),
    ("I have completed mongodb setup", {"entities": [(7, 29, "Completed")]}),
    ("data is trained and tested to complete the first schedule of the project", {"entities": [(30, 38, "Completed")]}),
    ("weekly report is completed and out for verification and approval", {"entities": [(17, 34, "Completed")]}),
    ("Mark has acknowledged the completion status of the module", {"entities": [(9, 43, "Completed")]}),
    ("I've completed task1", {"entities": [(5, 19, "Completed")]}),
    ("I've completed my work", {"entities": [(5, 21, "Completed")]}),
    ("I've completed my task", {"entities": [(5, 21, "Completed")]}),
    ("I've completed mongodb setup", {"entities": [(5, 27, "Completed")]}),
    ("I completed that task", {"entities": [(2,20, "Completed")]}),

    ("I didn't work on task1", {"entities": [(0,13, "Todo")]}),
    ("I didn't work on task1 and task2", {"entities": [(0,13, "Todo")]}),
    ("I have to start the Neo4j data modelling", {"entities": [(2,39, "Todo")]}),
    ("This call is for pending task of last week", {"entities": [(17,29, "Todo")]}),
    ("I will do data cleansing", {"entities": [(2, 23, "Todo")]}),
    ("I will do data cruching", {"entities": [(2, 22, "Todo")]}),
    ("Code submission in git is pending", {"entities": [(19,32, "Todo")]}),
    ("final version of slides are yet to be completed by next monday", {"entities": [(28,47, "Todo")]}),
    ("We are almost done finishing our todos for this week", {"entities": [(7,18, "Todo")]}),
    ("We are almost done finishing our todos for this week", {"entities": [(33,38, "Todo")]}),
    ("We had lot of todos in this week but we completed all of them", {"entities": [(14,19, "Todo")]}),
    ("We had lot of todos in this week but we completed all of them", {"entities": [(40,19, "Completed")]}),
    ("Do we have reasons for those items we missed this week in our todos", {"entities": [(62,66, "Todo")]}),
    ("I didn't work on optimizing search engines", {"entities": [(0,41, "Todo")]}),
    ("I am yet to complete the mongodb setup", {"entities": [(5,20, "Todo")]}),
   
    ("I need a document on Neo4j", {"entities": [(2,25, "Issue")]}),
    ("I need a document on use story", {"entities": [(2,29, "Issue")]}),
    ("I need a advice for use cases", {"entities": [(2,28, "Issue")]}),
    ("I am unable to do this", {"entities": [(5,21, "Issue")]}),
    ("I had issues with my system", {"entities": [(6,26, "Issue")]}),
    ("I had issues with my desktop", {"entities": [(6,27, "Issue")]}),
    ("I had issues with my system I see error screen", {"entities": [(6,27, "Issue")]}),
    ("I hope all our production issues are listed and kept ready for this meet", {"entities": [(26,32, "Issue")]}),
    ("Linda has addressed our issue in deployment and has given a word to us", {"entities": [(24,29, "Issue")]}),
    ("May be we must try using docker to fix our system differences and issues", {"entities": [(35,38, "Issue")]}),
    ("I had issues with my task1", {"entities": [(6,25, "Issue")]}),
    ("May be we must try using docker to fix our system differences and issues", {"entities": [(66,71, "Issue")]}),
    ("I am unable to do it", {"entities": [(5,11, "Issue")]}),
    ("I am unable to connect", {"entities": [(5,11, "Issue")]}),
    ("I had issues with my laptop", {"entities": [(6,26, "Issue")]}),
    ("I need a report on use story", {"entities": [(2,27, "Issue")]}),
  
    ("Lets catch up for weekend", {"entities": [(18,25, "Irrelevant")]}),
    ("See you then Bye", {"entities": [(13,15, "greet")]}),
    ("Hello am I audible", {"entities": [(0,5, "greet")]}),
    ("Hello. Yes you are", {"entities": [(0,5, "greet")]}),
    ("We are having this call for information systems module", {"entities": [(28,53, "Todo topic-technical")]}),
    ("Lets divide the work so that we integrate the codes quick", {"entities": [(32,50, "Todo")]}),
    ("I will do data modelling of the collected data as I am more comfortable with that part ", {"entities": [(10,24, "Todo")]}),
    ("I will do web scraping of the data in python using beautiful soup", {"entities": [(10,22, "Todo")]}),
    ("I will research on the use cases and the databases through which we can achieve them", {"entities": [(7,14, "Todo")]}),
    ("I already completed the mongodb initial setup", {"entities": [(10,44, "Completed")]}),
    ("can you please help me in that as I am unable to finish the initial setup", {"entities": [(39,72, "Issue")]}),
    ("I have completed reddis assignment which has to be submitted day after tomorrow", {"entities": [(8,35, "Completed")]}),
    ("can you please help me in that as I am unable to finish the initial setup", {"entities": [(39,72, "issue")]}),
    ("I already completed the mongodb initial setup", {"entities": [(10,44, "Completed")]}),
    ("I made a document on use cases", {"entities": [(3,30, "Completed")]}),
    ("seems like Mark loves this project more than family", {"entities": [(45,50, "Irrelevant")]}),
    ("We all deserve a bit fat meal this friday", {"entities": [(21,29, "Irrelevant")]}),
    ("Any one in our team having their birthday this month", {"entities": [(33,41, "Irrelevant")]}),
    ("I have my birthday this week", {"entities": [(10,18, "Irrelevant")]}),
    ("let's go to mall", {"entities": [(0,8, "Irrelevant")]}),
    ("We will go to mall", {"entities": [(0,10, "Irrelevant")]}),
    ("We'll go to mall", {"entities": [(0,8, "Irrelevant")]}),
    ("We'll have lunch together", {"entities": [(11,16, "Irrelevant")]})
]

# Adding labels to the `ner`
for _, annotations in train_data:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(30):

    # shuufling examples  before every iteration
    random.shuffle(train_data)
    losses = {}
    # batch up the examples using spaCy's minibatch
    for batch in spacy.util.minibatch(train_data, size=2):
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
        # Update the model
            nlp.update([example], losses=losses, drop=0.3)
        #print("Losses", losses)

# Save the model to directory

# Load the saved model and predict
""" output_dir = Path('C:/Users/chait/Documents/meetingmin/trained/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir) """

#output of speech to text 
a_dataframe = pd.read_csv("output/text.txt",header=None)
#print(a_dataframe)
#print(a_dataframe[1]

#import ibmspeech
#fin = ibmspeech.Textibm()
#print(fin[1])

enty = []
def entity_pred():
    for i in a_dataframe[0]:
        #print(i)
        doc = nlp(i)
        ent = "Entities", [(ent.text, ent.label_) for ent in doc.ents]
        #print(ent)
        #enty.append(i)
        enty.append(ent)
            
    print(enty)
    return enty 

enty = []
def entity_pred_neo():
    for i in a_dataframe[0]:
        #print(i)
        doc = nlp(i)
        
        ent = [ent.label_ for ent in doc.ents]
        print(ent)
        if(ent==[]):
            ent=['Unassigned']
        ent = ''.join(str(en) for en in ent[0])
        
        enty.append(ent)
        enty.append(i)
    greet=[]
    Unassigned=[]
    Completed=[]
    Issue = []
    Irrelevant = []
    ToDo = []

    for i,val in enumerate(enty):
        if(val == 'greet'):
           greet.append(i)
        if(val == 'Unassigned'):
           Unassigned.append(i)           
        if(val == 'Completed'):
           Completed.append(i) 
        if(val == 'Issue'):
           Issue.append(i)  
        if(val == 'Irrelevant'):
           Irrelevant.append(i)        
        if(val == 'ToDo'):
           ToDo.append(i)     
    greet_val = []   
    Unassigned_val = []
    Completed_val = []
    Issue_val = []
    Irrelevant_val = []
    ToDo_val = []

    for i in greet:
        greet_val.append(enty[i+1])
    for i in Unassigned:
        Unassigned_val.append(enty[i+1])    
    for i in Completed:
        Completed_val.append(enty[i+1])
    for i in Issue:
        Issue_val.append(enty[i+1])
    for i in Irrelevant:
        Irrelevant_val.append(enty[i+1])     
    for i in ToDo:
        ToDo_val.append(enty[i+1])                            
    
    max_length = max(len(greet_val), len(Unassigned_val), len(Completed_val),len(Issue_val),len(Irrelevant_val))
    print(max_length)

    for i in range(max_length-len(greet_val)):
        greet_val.append("")
    for i in range(max_length-len(Unassigned_val)):
        Unassigned_val.append("")   
    for i in range(max_length-len(Completed_val)):
        Completed_val.append("")  
    for i in range(max_length-len(Issue_val)):
        Issue_val.append("") 
    for i in range(max_length-len(Irrelevant_val)):
        Irrelevant_val.append("")   
    for i in range(max_length-len(ToDo_val)):
        ToDo_val.append("")   
    print(greet_val)  
    print(Unassigned_val)    
    print(Completed_val)                             
    from datetime import datetime
    
    
    d = {'Completed':  Completed_val,'greet':greet_val,'Unassigned':Unassigned_val,'Issues':Issue_val}
    print(d)
    
    df = pd.DataFrame(data=d)

    data = df[['Completed','greet','Unassigned','Issues']]
    #data = data[data.meeting.notnull()]

    list_theme = ['Completed','greet','Unassigned','Issues']
    execution_commands = []

    data_base_connection = GraphDatabase.driver(uri="bolt://localhost:7687",auth=("****","*****"))
    session = data_base_connection.session()
    
    for i, val in enumerate(list_theme):
        neo4j_create_statement = "Create(Task_type:Task{name:'"+str(val)+"'})"
        execution_commands.append(neo4j_create_statement)
    for i in execution_commands:
        session.run(i)
    
    
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

        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
        l = []
        for i in df_topic_keywords:
            l.append(list(df_topic_keywords[i]))
  
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
        
    d = datetime.today().strftime('%Y-%m-%d')
   
    session.run("Create(date:d{name:'"+str(d)+"'})")  

    execution_commands = [] 
    execution_commands.append("MATCH(Task_type:Task),(date:d {name: '"+str(d)+"'}) MERGE (date)-[r:has_tasks]->(Task_type)")
    for i in execution_commands:
        session.run(i)  
    #print(enty)
    return(enty)
"""doc = entity.nlp(row[1])
output_dir = Path('C:/Users/chait/Desktop/Case_study_2/trained/')

print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents]) 
"""

if __name__ == '__main__':
    entity_pred_neo()
    entity_pred()
    

    
       

    #dict ={}
    #for i in enty:
    #    if(print(i[0]=="greet")):

        
        
