from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from training_Validation_Insertion import train_validation
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from predictionFromModel import prediction
#path="Training_Batch_files/"
path = "Prediction_Batch_files/"
def main():
     path = "Prediction_Batch_files/"

    
     ##train_valObj = train_validation(path) #object initialization

     #train_valObj.train_validation()#calling the training_validation function


     #trainModelObj = trainModel() #object initialization
     #trainModelObj.trainingModel() #training the model for the files in the table

     pred_valObj=pred_validation(path)#object initialization
     pred_valObj.prediction_validation()#calling the prediction_validation function
     pred = prediction(path) #object initialization

     # predicting for dataset present in database
     path = pred.predictionFromModel()
     #return Response("Prediction File created at %s!!!" % path)

    
  
    
if __name__=='__main__':
    main()    

'''os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

port = int(os.getenv("PORT",5001))
if __name__ == "__main__":
    app.run(port=port,debug=True)    '''
