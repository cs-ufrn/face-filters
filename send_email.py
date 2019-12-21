import json
import smtplib
import socket
from email.mime.multipart import MIMEMultipart 
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.header import Header

import mimetypes

# Discription : Class Email

class Email:
    
    # Discription: function __init__ to inicialize our object Email. Constructor of Email.
    # Synopsis : __init__(self, conf_file)
    # Parameters : conf_file.
    # return value : none.
    
    def __init__(self, conf_file):
        self.client = None
        self.user = None
        password = None

        try:
            with open(conf_file) as config:
                data = json.load(config)
                self.user = data['address']
                password = data['password']
        except KeyError as e:
            print("Failed to load email configuaration", e)
        except FileNotFoundError as e:
            print("Email configuaration file not found ({})".format(e))


        if self.user and password:
            try:
                self.client = smtplib.SMTP('smtp.gmail.com:587', timeout=3)
                print('server ok!') 
                self.client.ehlo() # Can be omitted
                self.client.starttls() # Secure the connection
                self.client.ehlo() # Can be omitted

                self.client.login(self.user, password)
                print("successfully logged in")
            except smtplib.SMTPException as e:
                print("Error connecting:", e)
            except socket.timeout as e:
                print("Connection timed out:", e)
                
                
    # Discription : this function is to quit hte aplication.
    # Synopsis : __del__(self).
    # Parameters : none.
    # return value : none.

    def __del__(self):
        if self.client:
            try:
                self.client.quit()
            except smtplib.SMTPServerDisconnected:
                pass

    # Discription : This function e going to send an e-mail. 
    # Synopsis : send_email(self, attachments, receiver).
    # Parameters : attachments. receiver.
    # return value : depending on whether or not the function runs, a successful or an error message will be returned

    def send_email(self, attachments, receiver):
        if not self.client:
            no_client = "Email client not initialized correctly"
            print(no_client)
            return no_client

        try:
            mail = Email._make_mime("IEEE CS - UFRN", receiver, "CS face-filters",
                "Requested image attached!", attachments)
            self.client.sendmail(self.user, receiver, mail.as_string())
            print("sent successfully!")
        except smtplib.SMTPException as e:
            print("error connecting: ", e)
            return str(e)
        else:
            return "Email sent successfully!"

    # Discription : This function takes a path and attach this path to msg in the head. 
    # Synopsis : _get_attach_msg(path).
    # Parameters : path.
    # return value : It returns a message. 

    def _get_attach_msg(path):
        fp = open(path, 'rb')
        msg = MIMEImage(fp.read())
        fp.close()
        # Set the filename parameter
        msg.add_header('Content-Disposition', 'attachment', filename=path.split('/')[-1])
        return msg

    # Hehe

    def _make_mime(mail_from, mail_to, subject, body, attach_path):
        '''create MIME'''
        msg = MIMEMultipart()
        msg.set_charset
        msg['From'] = Header(mail_from, 'ascii')
        msg['To'] = Header(mail_to,  'ascii')
        msg['Subject'] = Header(subject, 'ascii')
        msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
        msg.attach(MIMEText(body,'plain')) 
        msg.attach(Email._get_attach_msg(attach_path))
        return msg