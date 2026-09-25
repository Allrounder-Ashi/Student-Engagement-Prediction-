def get_topic():

    while True:
        topic = input("\nEnter today's learning topic: ").strip()

        if topic:
            return topic

        print("Topic cannot be empty. Please enter a valid topic.")