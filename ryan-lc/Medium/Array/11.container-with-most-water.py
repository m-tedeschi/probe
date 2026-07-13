"""
11. Container With Most Water
Difficulty: Medium
https://leetcode.com/problems/container-with-most-water/

──────────────────────────────────────────────────

You are given an integer array height of length n. There are n
vertical lines drawn such that the two endpoints of the i^th line are
(i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such
that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array
[1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue
section) the container can contain is 49.

Example 2:

Input: height = [1,1]
Output: 1

 

Constraints:

	• n == height.length

	• 2 <= n <= 10^5

	• 0 <= height[i] <= 10^4
"""

class Solution:
    def maxArea(self, height: List[int]) -> int:
        # Time: O(n), Space: O(1)
        # Idea: Two pointers move inward, always discarding the shorter wall
        left = 0
        right = len(height) - 1
        max_area = 0

        while left < right:
            area = 0

            if height[left] < height[right]:
                area = (right - left) * height[left] # width * height
                left += 1
            else:
                area = (right - left) * height[right] # width * height
                right -= 1
            
            max_area = max(max_area, area)

        return max_area



