#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 03:16:20 2018

@author: yliu2
"""
#import machineLearning

from firebase import firebase
firebase = firebase.FirebaseApplication('https://zircobrowser-master.firebaseio.com', None)

#def main():
while 1==1:
  result = firebase.get('/', None)
  #result = firebase.get('/-L7xYdBkT3DSrcvac4UV', None)
  #print(result)
  #print(result.keys())
  #print('1')

  if result:
      print("The numebr of data records:", len(result.keys()))

      while (len(result.keys()) >= 200):
          features = list(result.values())
          # print(features)
          firebase.delete('/', None)

          import csv

          keys = features[0].keys()
          with open('testFirebase.csv', 'w') as output_file:
              # clear the existing content in csv file
              output_file.truncate()
              dict_writer = csv.DictWriter(output_file, keys)
              dict_writer.writeheader()
              dict_writer.writerows(features)



          # execute the ML models
          print('Testing with Machine Learning...')
          import machineLearning

          machineLearning.main()

          result = firebase.get('/', None)
          while (not result):
              print('No data')
              result = firebase.get('/', None)


  #main()

#main()


