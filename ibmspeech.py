import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from numpy.core.fromnumeric import size
import pandas as pd
from nltk import word_tokenize, pos_tag

authenticator = IAMAuthenticator('******')
service = SpeechToTextV1(authenticator=authenticator)

def getText():
    IBM_API_ENDPOINT = "*******"
    service.set_service_url(IBM_API_ENDPOINT)
    models = service.list_models().get_result()
    #print(json.dumps(models, indent=2))

    model = service.get_model('en-US_BroadbandModel').get_result()
    #print(json.dumps(model, indent=2))

    with open(join(dirname('__file__'), 'Audio/Audio1.wav'),
          'rb') as audio_file:
#    print(json.dumps(
        output = service.recognize(
        audio=audio_file,
        speaker_labels=True,
        content_type='audio/wav',
    #timestamps=True,
    #word_confidence=True,
        model='en-US_NarrowbandModel',
        continuous=True).get_result(),
        indent=2
    df = pd.DataFrame([i for elts in output for alts in elts['results'] for i in alts['alternatives']])
    d=[]
    for i in enumerate(output[0]['speaker_labels']):
            d.append(i[1]['speaker'])
    return (df,d)        

def replace(l,df):
    l1 = l  # use a list
    v = l1[0]                               # first char is our start v
        
    for idx,value in enumerate(l1[1:],1):   # we do all the others starting at index 1
        if l1[idx] == v:
            l1[idx] = '999'                  # replace list element
        else:
            v = l1[idx]                     # or replace v if different 

    l.append(l1)           # only create one string & convert to int 
    list_1 = [item for item in l if item!='999']
    l_updated = list_1[:size(df)]
    return list_1

def Textibm():
    df,d = getText()
    list_1 = replace(d,size(df))
    df['transcript']
    f = open('output/Text.txt', 'r+')
    f.truncate(0)
    df['transcript'].to_csv(r'output/Text.txt', header=None, index=None, sep=' ', mode='a')
    final = list(zip(list_1,df['transcript']))
    fin = pd.DataFrame(final)
    finsummary = pd.DataFrame(final)

    def determine_tense_input(sentence):
        text = word_tokenize(sentence)
        tagged = pos_tag(text)

        tense = {}
        tense["future"] = len([word for word in tagged if word[1] == "MD"])
        tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])  
        tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]]) 
        return(tense)
    data = []

    for i in fin[1]:
        data.append(determine_tense_input(i))
    #print(data)

    tense = []
    for i in range(0,len(data)):
        tense.append(max(data[i], key=data[i].get))
    #print(tense)    


    for i in range(0,len(data)):
        if(data[i]['present'] == data[i]['past'] == data[i]['future']):
            tense[i] = 'Unsure'
    #print(tense)

    for i in range(0,len(tense)):
        if(tense[i]=='past'):
            tense[i] = '---Completed'
        if(tense[i]=='future'):
            tense[i] == '---Todo'

    finsummary.loc[fin[0] == 0, 0] = "Person0: "
    finsummary.loc[fin[0] == 1, 0] = "Person1: "

    summary = finsummary[0]+finsummary[1]
    summary = summary.str.cat(sep='')
    f = open('output/summary.txt', 'r+')
    f.truncate(0)
    #print(summary)
    with open("output/summary.txt", "w") as text_file:
        text_file.write(summary)
    tense = pd.DataFrame(tense)
    #print(fin)
    return(fin)

if __name__=='__main__':
    Textibm()
