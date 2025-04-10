
def get_response(trial_count=None, max_count=None):
    while True:
        if trial_count is not None and max_count is not None:
            response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trial_count + 1}/{max_count})")
        else:
            response = input(f"Which way did the sound move? <----- 1 | 2 ----->")
        if response == "49" or response == "1":
            response = "left"
            break
        elif response == "51" or response == "2":
            response = "right"
            break
        else:
            continue
    return response
