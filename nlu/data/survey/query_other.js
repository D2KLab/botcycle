messages = db.getCollection('messages')
// TODO fill this
test_users = []
messages_filtered = test_users.map(el => {
    return {
        messages: messages.find({chat_id: el}).toArray().map(msg => {
            return {
                text: msg.text || null,
                type: msg.type || null,
                time: msg.time && msg.time.toTimeString()
            }}),
        user_id: el
    }})