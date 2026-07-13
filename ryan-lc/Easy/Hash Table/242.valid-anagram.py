"""
242. Valid Anagram
Difficulty: Easy
https://leetcode.com/problems/valid-anagram/

──────────────────────────────────────────────────

Given two strings s and t, return true if t is an anagram of s, and
false otherwise.

 

Example 1:

Input: s = "anagram", t = "nagaram"

Output: true

Example 2:

Input: s = "rat", t = "car"

Output: false

 

Constraints:

	• 1 <= s.length, t.length <= 5 * 10^4

	• s and t consist of lowercase English letters.

 

Follow up: What if the inputs contain Unicode characters? How would
you adapt your solution to such a case?
"""

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        char_freq = {}

        for ch in s:
            char_freq[ch] = char_freq.get(ch, 0) + 1

        for ch in t:
            char_freq[ch] = char_freq.get(ch, 0) - 1

            if char_freq[ch] < 0:
                return False

        """
        for key in char_freq:
            if char_freq[key] != 0:
                return False
        """
        
        return True

solve = Solution()
result = solve.isAnagram("racecar", "racecar")
print("Result: ", result)
