# WhatsApp call webhook (Meta → LiveKit)

Small Netlify function that turns inbound WhatsApp voice calls into LiveKit rooms.

```
Caller → WhatsApp Business number
       → Meta `calls` webhook (SDP offer)
       → this function (signature check)
       → LiveKit Connector: AcceptWhatsAppCall
       → LiveKit room + agent dispatch
```

## Endpoint

```
https://<site>.netlify.app/.netlify/functions/whatsapp
```

`GET` answers Meta's verification handshake, `POST` handles call events.

## Environment variables

| Variable | Description |
| --- | --- |
| `WHATSAPP_VERIFY_TOKEN` | Any string. Must match the Meta webhook "Verify token". |
| `WHATSAPP_APP_SECRET` | Meta app secret (App settings → Basic). Used to verify `X-Hub-Signature-256`. |
| `WHATSAPP_PHONE_NUMBER_ID` | WhatsApp phone number ID from the Meta app dashboard. |
| `WHATSAPP_ACCESS_TOKEN` | Permanent system-user token with `whatsapp_business_messaging`. |
| `WHATSAPP_CLOUD_API_VERSION` | Default `23.0`. Must match the version subscribed in Meta. |
| `LIVEKIT_URL` | `wss://<project>.livekit.cloud` |
| `LIVEKIT_API_KEY` | LiveKit API key. |
| `LIVEKIT_API_SECRET` | LiveKit API secret. |
| `LIVEKIT_AGENT_NAME` | Defaults to `MetaByte Solutions Receptionist`. |

## Deploy

```bash
cd webhook
npm install
npx netlify-cli deploy --prod
```

Then in Meta: **App → WhatsApp → Configuration**

- **Callback URL**: `https://<site>.netlify.app/.netlify/functions/whatsapp`
- **Verify token**: value of `WHATSAPP_VERIFY_TOKEN`
- Subscribe to the **`calls`** field (same Cloud API version as `WHATSAPP_CLOUD_API_VERSION`).

## Local testing

```bash
npx netlify-cli dev
curl "http://localhost:8888/.netlify/functions/whatsapp?hub.mode=subscribe&hub.verify_token=$WHATSAPP_VERIFY_TOKEN&hub.challenge=123"
```
