Attribute VB_Name = "Module1"
Sub ClearBoard()
' Clear board range, macro for clear button

    Range("board") = 0
End Sub

Sub FlattenRange()
' Take the board range and reference it into a single column

    Dim i As Integer
    i = 0
    For Each cell In Range("Table4")
        Selection.Offset(i, 0).Value = "=ReLU!" & cell.Address
        i = i + 1
    Next cell
End Sub

Sub Label()
' Label each cell in board range with number
    
    Dim i As Integer
    i = 1
    For Each cell In Range("board")
        cell.Value = i
        i = i + 1
    Next cell
End Sub
