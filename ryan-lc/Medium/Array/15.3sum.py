"""
15. 3Sum
Difficulty: Medium
https://leetcode.com/problems/3sum/

──────────────────────────────────────────────────

Given an integer array nums, return all the triplets [nums[i],
nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] +
nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets
does not matter.

Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.

 

Constraints:

	• 3 <= nums.length <= 3000

	• -10^5 <= nums[i] <= 10^5
"""

class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums_sorted = sorted(nums)
        triplets = []
        prev = float('-inf')

        for i in range(len(nums_sorted) - 2):
            if nums_sorted[i] == prev:
                continue

            prev = nums_sorted[i]
            left = i + 1
            right = len(nums_sorted) - 1

            while left < right:
                total = nums_sorted[i] + nums_sorted[left] + nums_sorted[right]

                if total < 0:
                    left += 1
                    continue
                if total > 0:
                    right -= 1
                    continue

                triplets.append([nums_sorted[i], nums_sorted[left], nums_sorted[right]])

                left += 1
                right -= 1

                while left < right and nums_sorted[left - 1] == nums_sorted[left]:
                    left += 1

                while left < right and nums_sorted[right + 1] == nums_sorted[right]:
                    right -= 1

        return triplets
