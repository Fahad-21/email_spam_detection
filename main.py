import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import os
import pickle
# Gmail API utils
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
# for encoding/decoding messages in base64
import base64
from base64 import urlsafe_b64decode, urlsafe_b64encode

spam_mails = {}
ids = []
ready_to_delete = False

def Model(mail, id):
    global spam_mails
    # Data PreProcessing
    # loading the data from csv file to pandas Dataframe.
    raw_mail_data = pd.read_csv('dataset.csv')

    #print(raw_mail_data)

    # replacing the null values with a nulll string.

    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

    # printing the first 5 rows of the dataframe

    #print(mail_data.head())

    # checking number of rows and columns in dataframe
    #print(mail_data.shape)

    #labeling spam mail as 0; and ham mail as 1;

    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

    #seperating the data as texts and lebel

    X = mail_data['Message']
    Y = mail_data['Category']

    #print(X)
    #print(Y)

    # Training the Model.

    # Splitting the data into training data and testing data.

    X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    #print(X.shape)
    #print(X_train.shape)
    #print(X_test.shape)

    #Feature Extraction

    # transform the test data to feature vectors that can be used as input to the Logistic Regression
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    #convert Y_train and Y_test values as integers

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    #print(X_train)
    #print(X_train_features)

    model = LogisticRegression()

    # training the Logistic Regression model with the training data

    model.fit(X_train_features, Y_train)

    # prediction on training data
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    print("Accuracy on traininf data: ", accuracy_on_training_data)

    # prediction on test data

    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    print("Accuracy on predictionf data: ", accuracy_on_training_data)

    #building  predictive system

    input_mail = [mail]

    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction

    prediction = model.predict(input_data_features)
    #print(prediction)


    if (prediction[0]==1):
      print('Ham mail')

    else:
      #spam_mails[str(id)] = mail
      ids.append(id)
      print('Spam mail')
      return True

# Request all access (permission to read/send/receive emails, manage the inbox, and more)
SCOPES = ['https://mail.google.com/']
our_email = 'tenmax054@gmail.com'

# authentication
def gmail_authenticate():
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)



# get the Gmail API service
service = gmail_authenticate()


# Reading emails with specific content.

# utility functions

def search_messages(service, query):
    result = service.users().messages().list(userId='me',q=query).execute()
    messages = [ ]
    if 'messages' in result:
        messages.extend(result['messages'])
    while 'nextPageToken' in result:
        page_token = result['nextPageToken']
        result = service.users().messages().list(userId='me',q=query, pageToken=page_token).execute()
        if 'messages' in result:
            messages.extend(result['messages'])
    return messages

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


def clean(text):
    # clean text for creating a folder
    return "".join(c if c.isalnum() else "_" for c in text)

def parse_parts(service, parts, folder_name, message):
    """
    Utility function that parses the content of an email partition
    """

    if parts:
       # print('a')
        for part in parts:
        #    print('b')
            filename = part.get("filename")
            mimeType = part.get("mimeType")
            body = part.get("body")
            data = body.get("data")
            file_size = body.get("size")
            part_headers = part.get("headers")
            if part.get("parts"):
                #print('c')
                # recursively call this function when we see that a part
                # has parts inside
                parse_parts(service, part.get("parts"), folder_name, message)
            if mimeType == "text/plain":
                #print('d')
                # if the email part is text plain
                if data:
                    #print('e')
                    text = urlsafe_b64decode(data).decode()
                    #print(text)
            elif mimeType == "text/html":
                #print('f')
                # if the email part is an HTML content
                # save the HTML file and optionally open it in the browser
                if not filename:
                    filename = "index.html"
                filepath = os.path.join(folder_name, filename)
                #print("Saving HTML to", filepath)
                with open(filepath, "wb") as f:
                    f.write(urlsafe_b64decode(data))
            else:
                # attachment other than a plain text or HTML
                for part_header in part_headers:
                    part_header_name = part_header.get("name")
                    part_header_value = part_header.get("value")
                    if part_header_name == "Content-Disposition":
                        if "attachment" in part_header_value:
                            # we get the attachment ID
                            # and make another request to get the attachment itself
                            #print("Saving the file:", filename, "size:", get_size_format(file_size))
                            attachment_id = body.get("attachmentId")
                            attachment = service.users().messages() \
                                        .attachments().get(id=attachment_id, userId='me', messageId=message['id']).execute()
                            data = attachment.get("data")
                            filepath = os.path.join(folder_name, filename)
                            if data:
                                with open(filepath, "wb") as f:
                                    f.write(urlsafe_b64decode(data))

def read_message(service, message):
    """
    This function takes Gmail API `service` and the given `message_id` and does the following:
        - Downloads the content of the email
        - Prints email basic information (To, From, Subject & Date) and plain/text parts
        - Creates a folder for each email based on the subject
        - Downloads text/html content (if available) and saves it under the folder created as index.html
        - Downloads any file that is attached to the email and saves it in the folder created
    """
    # msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
    msg = service.users().messages().get(userId='me', id=message['id']).execute()
    # parts can be the message body, or attachments
    payload = msg['payload']
    # print("Payload", payload)
    print('id', message['id'])
    headers = payload.get("headers")
    try:
        parts = payload.get("parts")[0]
        data = parts['body']['data']
        data = data.replace("-", "+").replace("_", "/")
        real_message = base64.urlsafe_b64decode(data).decode('utf-8')
        r_m = real_message.replace('\n', '')
        result = Model(r_m, message['id'])
        if result == True:
            service.users().messages().delete(userId='me', id=message['id']).execute()
            print("message deleted")
    except:
        real_message = msg.get("payload").get("body").get("data")
        r_m = base64.urlsafe_b64decode(real_message).decode('utf-8')
        result = Model(r_m, message['id'])
        if result == True:
            service.users().messages().delete(userId='me', id=message['id']).execute()
            print("message deleted")
        pass


   # real_message = msg.get("payload").get("body").get("data")

    #real_message = base64.urlsafe_b64decode(msg.get("payload").get("body").get("data").encode("ASCII")).decode("utf-8")
    #print("real", real_message)




    parts = payload.get("parts")
    folder_name = "email"
    #print(parts)

    has_subject = False
    if headers:
        # this section prints email basic info & creates a folder for the email
        for header in headers:
            # print("headert", header)
            name = header.get("name")
            value = header.get("value")

            if name.lower() == 'from':
                # we print the From address
     #           print("From:", value)
                pass
            if name.lower() == "to":
                # we print the To address
                pass
               # print("To:", value)
            if name.lower() == "subject":
                # make our boolean True, the email has "subject"

                has_subject = True
                # make a directory with the name of the subject
                folder_name = clean(value)
                # we will also handle emails with the same subject name
                folder_counter = 0
                while os.path.isdir(folder_name):
                    folder_counter += 1
                    # we have the same folder name, add a number next to it
                    if folder_name[-1].isdigit() and folder_name[-2] == "_":
                        folder_name = f"{folder_name[:-2]}_{folder_counter}"
                    elif folder_name[-2:].isdigit() and folder_name[-3] == "_":
                        folder_name = f"{folder_name[:-3]}_{folder_counter}"
                    else:
                        folder_name = f"{folder_name}_{folder_counter}"
                try:
                    os.mkdir(folder_name)
                except:
                    pass
                #print("Subject:", value)
            if name.lower() == "date":
                # we print the date when the message was sent
                #print("Date:", value)
                pass
    if not has_subject:
        # if the email does not have a subject, then make a folder with "email" name
        # since folders are created based on subjects
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
    parse_parts(service, parts, folder_name, message)
    print("="*50)



# get emails that match the query you specify
results = search_messages(service, "Free")
#print(results)
print(f"Found {len(results)} results.")
# for each email matched, read it (output plain/text to console & save HTML and attachments)
for msg in results:
    read_message(service, msg)

