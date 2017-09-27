# 
## Get required tokens

### Botkit token

The communication with messaging platforms is done talking to [botcycle-botkit](https://github.com/MartinoMensio/botcycle-botkit) over a websocket with an authentication token. This token must be placed in the `.env` file as `WEBSOCKET_TOKEN`.

### Wit.ai token

The token for `wit.ai` is linked to the online model. You can create your own token but the classifier that is used has been configured online (entities and intent) and trained with some sentences.

An *export data* exists on `wit.ai` in order to download all the data (expressions and entities). Can be useful.

The `WIT_TOKEN` token has to be stored in the `.env` file.

### Google maps API token

This app uses google maps API for geocoding. The environment variable `MAPS_TOKEN` needs to be provided too.
