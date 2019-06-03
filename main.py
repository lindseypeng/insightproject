#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:13:37 2019

@author: lindsey
"""

from flask import Flask, render_template, request
from bert_emb_modul import mastercode as ms
import numpy as  np
# Create the application object
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def tag_output():
#       
       # Pull input
       some_input =request.args.get('user_input')     
       print("______________________________________________________")
       print(some_input)
       ##requests.forms.get
       # Case if empty
       if some_input == '':
           return render_template("index.html",
                                  my_input = some_input,
                                  my_form_result="Empty")
       else:
           outputinit=ms(some_input)
           yoursentence=outputinit.output_neighbors()
           some_output=yoursentence
           some_output2=3
           some_image="myfigure.png"
           return render_template("index.html",
                              my_input=some_input,
                              my_output=some_output,
                              my_number=some_output2,
                              my_img_name=some_image,
                              my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
