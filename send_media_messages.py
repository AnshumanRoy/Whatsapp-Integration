# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client
from decouple import config


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = config("TWILIO_ACCOUNT_SID")
auth_token = config("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

message = client.messages \
    .create(
         media_url=['https://images.unsplash.com/photo-1545093149-618ce3bcf49d?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=668&q=80'],
         from_="whatsapp:"+config("TWILIO_NUMBER"),
         to='whatsapp:+917225830543'
     )

print(message.media._uri)
