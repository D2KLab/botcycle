messages = db.getCollection('messages')
test_users = db.getCollection('users').find().toArray()
messages_filtered = test_users.map(el => {
    return {
        messages: messages.find({chat_id: el._id}).toArray().map(msg => {
            return {
                text: msg.text || null,
                type: msg.type || null,
                time: msg.time.toTimeString()
            }
        }),
        user_id: el._id
    }
})