# def getMaxRequests(bandwidth, requests, total_bandwidth):
#     n = len(bandwidth)
#
#     # Create a 2D list to store the maximum requests for each combination of endpoints and bandwidth
#     dp = [[0] * (total_bandwidth + 1) for _ in range(n + 1)]
#
#     for i in range(1, n + 1):
#         for j in range(total_bandwidth + 1):
#             # Check if the current endpoint's bandwidth can fit into the available bandwidth
#             if bandwidth[i - 1] <= j:
#                 # Compare the maximum requests with and without considering the current endpoint
#                 dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - bandwidth[i - 1]] + requests[i - 1])
#             else:
#                 dp[i][j] = dp[i - 1][j]
#
#     return dp[n][total_bandwidth]
#
# # Example usage:
# n = 5
# total_bandwidth = 300
# bandwidth = [200, 100, 350, 50, 100]
# requests = [270, 142, 450, 124, 189]
#
# result = getMaxRequests(bandwidth, requests, total_bandwidth)
# print("Maximum total number of requests:", result)


from collections import defaultdict

def message_delivery(n, timestamps, messages, k):
    last_delivery_time = defaultdict(lambda: float('-inf'))
    result = []

    for i in range(n):
        current_time = timestamps[i]
        current_message = messages[i]

        if current_time - last_delivery_time[current_message] >= k:
            result.append(True)
            last_delivery_time[current_message] = current_time
        else:
            result.append(False)

    return result

# Example usage:
n = 6
timestamps = [1, 4, 5, 10, 11, 14]
messages = ["hello", "bye", "bye", "hello", "bye", "hello"]
k = 5

result = message_delivery(n, timestamps, messages, k)
print(result)
