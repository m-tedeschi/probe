from collections import deque
from heapq import heappop, heappush
from typing import List

"""
2812. Find the Safest Path in a Grid
Difficulty: Medium
https://leetcode.com/problems/find-the-safest-path-in-a-grid/

──────────────────────────────────────────────────

You are given a 0-indexed 2D matrix grid of size n x n, where (r, c)
represents:

	• A cell containing a thief if grid[r][c] = 1

	• An empty cell if grid[r][c] = 0

You are initially positioned at cell (0, 0). In one move, you can
move to any adjacent cell in the grid, including cells containing
thieves.

The safeness factor of a path on the grid is defined as the minimum
manhattan distance from any cell in the path to any thief in the grid.

Return the maximum safeness factor of all paths leading to cell (n -
1, n - 1).

An adjacent cell of cell (r, c), is one of the cells (r, c + 1), (r,
c - 1), (r + 1, c) and (r - 1, c) if it exists.

The Manhattan distance between two cells (a, b) and (x, y) is equal
to |a - x| + |b - y|, where |val| denotes the absolute value of val.

 

Example 1:

Input: grid = [[1,0,0],[0,0,0],[0,0,1]]
Output: 0
Explanation: All paths from (0, 0) to (n - 1, n - 1) go through the
thieves in cells (0, 0) and (n - 1, n - 1).

Example 2:

Input: grid = [[0,0,1],[0,0,0],[0,0,0]]
Output: 2
Explanation: The path depicted in the picture above has a safeness
factor of 2 since:
- The closest cell of the path to the thief at cell (0, 2) is cell
(0, 0). The distance between them is | 0 - 0 | + | 0 - 2 | = 2.
It can be shown that there are no other paths with a higher safeness
factor.

Example 3:

Input: grid = [[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]
Output: 2
Explanation: The path depicted in the picture above has a safeness
factor of 2 since:
- The closest cell of the path to the thief at cell (0, 3) is cell
(1, 2). The distance between them is | 0 - 1 | + | 3 - 2 | = 2.
- The closest cell of the path to the thief at cell (3, 0) is cell
(3, 2). The distance between them is | 3 - 3 | + | 0 - 2 | = 2.
It can be shown that there are no other paths with a higher safeness
factor.

 

Constraints:

	• 1 <= grid.length == n <= 400

	• grid[i].length == n

	• grid[i][j] is either 0 or 1.

	• There is at least one thief in the grid.
"""

class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        dist = [[-1] * n for _ in range(n)]
        queue = deque()

        for row in range(n):
            for col in range(n):
                if grid[row][col] == 1:
                    dist[row][col] = 0
                    queue.append((row, col))

        while queue:
            row, col = queue.popleft()
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[row][col] + 1
                    queue.append((nr, nc))

        best = [[-1] * n for _ in range(n)]
        best[0][0] = dist[0][0]
        heap = [(-dist[0][0], 0, 0)]

        while heap:
            safeness, row, col = heappop(heap)
            safeness = -safeness

            if row == n - 1 and col == n - 1:
                return safeness
            if safeness < best[row][col]:
                continue

            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n:
                    next_safeness = min(safeness, dist[nr][nc])
                    if next_safeness > best[nr][nc]:
                        best[nr][nc] = next_safeness
                        heappush(heap, (-next_safeness, nr, nc))

        return 0
