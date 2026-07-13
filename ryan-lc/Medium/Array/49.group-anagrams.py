"""
49. Group Anagrams
Difficulty: Medium
https://leetcode.com/problems/group-anagrams/

──────────────────────────────────────────────────

Given an array of strings strs, group the anagrams together. You can
return the answer in any order.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

	• There is no string in strs that can be rearranged to form "bat".

• The strings "nat" and "tan" are anagrams as they can be rearranged
to form each other.

• The strings "ate", "eat", and "tea" are anagrams as they can be
rearranged to form each other.

Example 2:

Input: strs = [""]

Output: [[""]]

Example 3:

Input: strs = ["a"]

Output: [["a"]]

 

Constraints:

	• 1 <= strs.length <= 10^4

	• 0 <= strs[i].length <= 100

	• strs[i] consists of lowercase English letters.
"""

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = [] # Result list
        seen_sorted = {}

        for word in strs:
            # Alphabetically sort the word's characters, then join them into a
            # string (string -> list -> string)
            key = "".join(sorted(word))

            # If we've not seen this sorted string before, it becomes a new key
            # and we store the initial word it corresponds to in it
            if not key in seen_sorted:
                seen_sorted[key] = [word]
            else: # Otherwise we've seen this key before, and can append the word to it
                seen_sorted[key].append(word)

        # Construct the final result list from our built hash map
        for val in seen_sorted.values():
            anagrams.append(val)

        return anagrams
