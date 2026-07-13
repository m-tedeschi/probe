"""
1288. Remove Covered Intervals
Difficulty: Medium
https://leetcode.com/problems/remove-covered-intervals/

──────────────────────────────────────────────────

Given an array intervals where intervals[i] = [li, ri] represent the
interval [li, ri), remove all intervals that are covered by another
interval in the list.

The interval [a, b) is covered by the interval [c, d) if and only if
c <= a and b <= d.

Return the number of remaining intervals.

 

Example 1:

Input: intervals = [[1,4],[3,6],[2,8]]
Output: 2
Explanation: Interval [3,6] is covered by [2,8], therefore it is
removed.

Example 2:

Input: intervals = [[1,4],[2,3]]
Output: 1

 

Constraints:

	• 1 <= intervals.length <= 1000

	• intervals[i].length == 2

	• 0 <= li < ri <= 10^5

	• All the given intervals are unique.
"""

class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        # Sorts by first element (not sure why -x[1])
        intervals.sort(key=lambda x: (x[0], -x[1])) 
        # Initialize results list 
        result = [intervals[0]] 

        # Iterate through each interval
        for left, right in intervals:
            # Unpack the last result interval
            prevLeft, prevRight = result[-1]

            # If the interval isn't covered, we can skip it
            if prevLeft <= left and prevRight >= right:
                continue

            # Otherwise, we must append it to the results list
            result.append([left, right])

        return len(result)
