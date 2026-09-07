pragma ComponentBehavior: Bound
import QtQuick

/*!
    FocusRing - the 2 px keyboard-focus ring (accent at 60 %, solid palette
    highlight in high contrast).  Wraps `target` (default: the parent) with an
    `outset` margin so no control carries a resting hairline; only visible while
    `active`.
*/
Rectangle {
    id: ring

    property bool active: false
    property Item target: parent
    property real ringRadius: Theme.radiusControl
    property real outset: 2

    readonly property bool onParent: target === parent
    x: (onParent ? 0 : target.x) - outset
    y: (onParent ? 0 : target.y) - outset
    width: (target ? target.width : 0) + 2 * outset
    height: (target ? target.height : 0) + 2 * outset
    radius: ringRadius + outset
    color: "transparent"
    border.width: 2
    border.color: Theme.focusColor
    antialiasing: true
    visible: active
    z: 10
}
