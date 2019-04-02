import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.header import Header

import mimetypes

def _get_attach_msg(path):
    fp = open(path, 'rb')
    msg = MIMEImage(fp.read())
    fp.close()
    # Set the filename parameter
    msg.add_header('Content-Disposition', 'attachment', filename=path.split('/')[-1])
    return msg

def _make_mime(mail_from, mail_to, subject, body, attachments):
    '''create MIME'''
    msg = MIMEMultipart()
    msg['From'] = Header(mail_from, 'ascii')
    msg['To'] = Header(mail_to,  'ascii')
    msg['Subject'] = Header(subject, 'ascii')
    msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
    msg.attach( MIMEText(body,'plain')) 
    msg.attach(_get_attach_msg(attachments))
    return msg


def send_email(attachments,receiver):
    #pegar credenciais
    #user = 
    #password = 

    try:
        client = smtplib.SMTP('smtp.gmail.com:587')
        print('server ok!') 
        client.ehlo() # Can be omitted
        client.starttls() # Secure the connection
        client.ehlo() # Can be omitted
        client.login('''user''','''password''' )
        client.sendmail("no.reply.face.filters@gmail.com", receiver, _make_mime("CS - IEEE - UFRN",receiver, "cs-facefilters", "Download your file", attachments).as_string())
        client.close()
        print("sent successfully!")
    except smtplib.SMTPException:
    	print("error connecting...")