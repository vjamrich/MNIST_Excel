VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet3"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Private Sub Worksheet_SelectionChange(ByVal Target As Range)
    
    Set isect = Application.Intersect(Range("board"), Selection)
    
    If Draw.Value = True Then
        If Not isect Is Nothing Then
            If Selection.Cells.Count <= 1 Then
                If Selection = 0 Then
                    Selection = 1
                Else
                    Selection = 0
                End If
            End If
        End If
    End If
    
End Sub
