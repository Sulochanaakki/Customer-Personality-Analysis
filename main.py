from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from training_Validation_Insertion import train_validation
from trainingModel import trainModel
path="Training_Batch_files/"
def main():
    
     train_valObj = train_validation(path) #object initialization

     train_valObj.train_validation()#calling the training_validation function


     trainModelObj = trainModel() #object initialization
     trainModelObj.trainingModel() #training the model for the files in the table


    
  
    
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
