# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:02:41 2023

@author: marcos
"""
# Upper Confidence Bound (UCB) Algorithm
# Select the best ad to show to a user based in the UCB
"""
The Upper Confidence Bound (UCB) algorithm is an effective approach for selecting the best ad to display to a user.
It balances exploration and exploitation by considering both the estimated reward and the uncertainty associated with each ad.
The algorithm initializes estimates and confidence bounds for each ad, and then updates them based on observed rewards.
This enables the algorithm to explore different ads initially and gradually shift towards exploiting the ones that appear to be more promising.
By continually updating estimates and confidence bounds, the UCB algorithm adapts over time and tends to converge towards selecting the best-performing ad more frequently.
This approach ensures that the advertising strategy maximizes its effectiveness by dynamically adjusting the selection process based on the available information.

Applying the UCB algorithm in the context of ad selection allows for adaptive decision-making.
The algorithm strikes a balance between exploring new options and exploiting the best-performing ads.
This ensures that the advertising campaign is not solely based on existing knowledge but actively seeks new insights.
The algorithm makes informed choices that gradually lead to the selection of the most effective ad for a given user.
Overall, the UCB algorithm is a powerful tool for optimizing ad selection and maximizing the overall success of advertising campaigns.
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
#                                     Main
# =============================================================================

np.random.seed(42)

# Define the number of ads to choose from
num_ads = 10

print("\nUCB - AD CAMPAIGN\n")
print("\nNúmero de Ads: " + str(num_ads))

# Initialize arrays to store the number of times each ad has been chosen
# and the sum of rewards for each ad
num_choices = np.zeros(num_ads)
reward_sums = np.zeros(num_ads)

# Define the number of rounds (i.e., the number of times to choose an ad)
num_rounds = 1000

num_choices_round = []
reward_sums_round = []

# Choose each ad once initially to ensure all ads are explored
for i in range(num_ads):
    # Choose the ad - 0 or 1
    reward = np.random.randint(10)

    # Update the number of times the ad has been chosen and the sum of rewards
    num_choices[i] += 1
    reward_sums[i] += reward

    num_choices_round.extend(num_choices)
    reward_sums_round.extend(reward_sums)


# Choose ads using the UCB algorithm for the remaining rounds
for i in range(num_ads, num_rounds):
    # constant value
    c = 7

    # Compute the upper confidence bound for each ad
    ucbs = reward_sums / num_choices + np.sqrt(c * np.log(i) / num_choices)

    # Choose the ad with the highest UCB
    ad = np.argmax(ucbs)

    # Reward for the ad
    reward = np.random.randint(10)

    # Update the number of times the ad has been chosen and the sum of rewards
    num_choices[ad] += 1
    reward_sums[ad] += reward

    num_choices_round.extend(num_choices)
    reward_sums_round.extend(reward_sums)


num_choices_round = [num_choices_round[i:i+num_ads] for i in range(0, len(num_choices_round), num_ads)]
reward_sums_round = [reward_sums_round[i:i+num_ads] for i in range(0, len(reward_sums_round), num_ads)]

plt.title("Reward per ad")
plt.plot(reward_sums_round)
plt.show()

plt.title("Selected ad")
plt.plot(num_choices_round)
plt.show()
