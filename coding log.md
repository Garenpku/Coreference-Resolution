**4.8**
- Implemented GRU version, where only a sequence encoding is applied to extract the features of a discourse.
- Fixed a bug: the target token position should not be subtracted by 1.
- Reexamined the evaluation part: substituted the predicted token with gold token and the F score approaches 1 (No bug here).

The new performance is still not satisfactory. Seems to be the problem of the model itself (or my implementation).
