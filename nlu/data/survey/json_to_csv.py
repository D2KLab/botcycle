import json
import csv

with open('json_others.json') as json_file:
  other = json.load(json_file)


with open('json_slack.json') as json_file:
  slack = json.load(json_file)

other.extend(slack)

with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['text', 'type', 'time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for user in other:
      for message in user['messages']:
        writer.writerow(message)