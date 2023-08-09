List of notable changes.

Controller:
- `ExpiringDict` was not synchronizing the timestamp heap and the user dict on `__delitem__`. Fixed.
- The user prompt was sent to the controller in two separate POST requests, but the controller was lacking synchronization over the user state. Fixed.
  - Especially, randomly assigning model names could be called twice.
  - This was addressed by switching to async code (which serializes access to the user state dictionary). This also allows FastAPI to optimize better (according to the docs).
- User ID to Request ID
  - Users may play the Colosseum multiple times, and all those runs sharing a user ID simply makes things complicated. We did talk about finding malicious users through shared user IDs, but this detection method can just be avoided by refreshing anyway.
  - `X-User-ID` was removed. Now, `ControllerClient`, a client implementation that allows the Gradio app to talk to the backend controller, is wrapped in a `gr.State`. `ControllerClient` overrides `__deepcopy__`, which creates a new UUID4 request ID and returns the new client that has the new request ID. `__deepcopy__` is invoked when (1) `gr.State` copies the client on a new session, and (2) when the user clicks the Clear button.
- `threading` to `asyncio.Task` -- The heartbeat task don't run every second of course, but sending a request to each worker involves across the network makes it a good candidate for an async IO task.
- Added incremental request state logging
  - Even if the user exits without voting, we still want to record at least part of the user's state (e.g., at least prompt, response, and energy, which will later turn out to be useful).
- Periodic heartbeat checks do not prevent a TGI server dying right after the check and the controller experiencing a connection error on `POST /generate_stream`.
  - Now, when the controller experiences a connection error, it immediately deactivates the worker. Then, the heartbeat task will periodically wake up and see if the dead TGI server is back up, and reinstate its worker state.

Gradio app:
- Added `queue=False` to event listeners that are not related to the Colosseum.
- Excessive code repetition due to the parallel structure of left and right votes
- Vote buttons being enabled after one model is done, instead of both models being done
- When the user clicks submit or presses enter, two parallel event handlers are launched. This made two things difficult: (1) removing the user's prompt from the input box (i.e., modifying the `TextBox` component) is a race condition that wasn't being handled (Until now there was only one user testing the app and the two initial handlers managed to both read the prompt, but when load increases, the handler that ran late may see an empty prompt text box because another handler deleted its content.) (2) enabling the response vote buttons only after both the models are done responding.
  - Ref: https://github.com/gradio-app/gradio/issues/3908
  - I consolidated generating the responses of two models into one event handler.
- The wordings of the final energy question changed. Now, it asks whether better response quality was worth the increase in energy consumption. I thought asking which model is more energy efficient could be confusing. I don't think there is a very obvious and unique definition of energy efficiency in our context.
- Autofocus the prompt textarea on reset.

Miscellaneous notes:
- I tried to see if I can write a Docker Compose file and deploy it with our Swarm cluster with `docker stack deploy`, but I wasn't able to get it working within a reasonable amount of time. GPUs being involved was the primary pain. So I just gave up and manually set up our stuff.
