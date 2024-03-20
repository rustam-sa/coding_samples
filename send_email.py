import smtplib
from email.message import EmailMessage

# set your email and password
# please use App Password
email_address = "rustamworksklb@gmail.com"
email_password = "pnyainhlqcjipdcn"

def send_email(subject, message):
    # create email
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = email_address
    msg['To'] = "rustam308@gmail.com"
    msg.set_content(message)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_address, email_password)
        smtp.send_message(msg)