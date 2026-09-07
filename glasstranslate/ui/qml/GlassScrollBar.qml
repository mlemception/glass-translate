pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    GlassScrollBar - a thin ink-coloured scroll indicator for the page
    Flickables and the combo popup list.  Attach with
    `T.ScrollBar.vertical: GlassScrollBar {}`.
*/
T.ScrollBar {
    id: control

    property string label: "Scroll"

    implicitWidth: 8
    implicitHeight: 8
    padding: 2
    minimumSize: 0.08
    policy: T.ScrollBar.AsNeeded
    interactive: true
    Accessible.role: Accessible.ScrollBar
    Accessible.name: label

    contentItem: Rectangle {
        implicitWidth: 4
        implicitHeight: 4
        radius: 2
        color: Theme.mode === "hc" ? Theme.sysPal.windowText
             : Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, control.pressed ? 0.55 : 0.32)
        opacity: control.policy === T.ScrollBar.AlwaysOn || (control.active && control.size < 1.0) ? 1 : 0
        Behavior on opacity { NumberAnimation { duration: Theme.fade } }
    }
}
