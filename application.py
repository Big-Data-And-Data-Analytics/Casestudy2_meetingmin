import os
from flask import Flask, render_template, send_file, request
import ibmspeech
import entity
#import entity_prediction
import summarisation
#import prac
#import prediction_pipeline
import pandas as pd
 
application = Flask(__name__)

@application.route('/')
def index():
    return render_template("index.html")

@application.route('/analysis',methods=['POST'])
def getAudio():
    f = request.files['audio_file']
    print('File uploaded successfully')

    with open('Audio/Audio1.wav','wb') as audio:
        f.save(audio)

    f = ibmspeech.Textibm()

    #f.to_html(classes='female')
    ent = entity.entity_pred()
    summ_text = summarisation.summ_pred()
    
    return render_template('summary.html',tables=[f.to_html(classes='data')], titles=f.columns.values,ent_text = ent,summ_text = summ_text)
   
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(debug=True, host='0.0.0.0', port=port)
    #app.run(debug= True)


