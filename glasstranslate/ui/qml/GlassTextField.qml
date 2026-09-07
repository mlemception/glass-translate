pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    GlassTextField - T.TextField (docs/GLASS_DESIGN.md §2.3): the "fill"
    material with no resting underline; the 2 px accent ring appears on
    activeFocus (keyboard or click).  `placeholder`, `password` (echo mode),
    `monospace` (Theme.mono).  `committed(text)` fires on Enter and on focus
    loss.  Setting `error = true` plays an ink-coloured shake (6 px, 3 cycles,
    300 ms; nothing under reduceMotion) and resets itself; the message belongs
    in the status strip.  `label` / `hint` come from the FormRow and feed
    Accessible.*.
*/
T.TextField {
    id: control

    property string label: ""
    property string hint: ""
    property string placeholder: ""
    property bool password: false
    property bool monospace: false
    property bool error: false
    property real radius: Theme.radiusControl
    signal committed(string text)

    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset,
                            contentWidth + leftPadding + rightPadding, 120)
    implicitHeight: Math.max(Theme.controlH, fm.height + 2 * Theme.padY,
                             contentHeight + topPadding + bottomPadding)
    leftPadding: 12
    rightPadding: 12
    topPadding: Theme.padY
    bottomPadding: Theme.padY
    font.family: monospace ? Theme.mono : Theme.family
    font.pointSize: Theme.fontBody
    color: inkColor
    selectionColor: Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.25)
    selectedTextColor: Theme.ink
    placeholderText: placeholder
    placeholderTextColor: Theme.inkSecondary
    echoMode: password ? TextInput.Password : TextInput.Normal
    selectByMouse: true
    verticalAlignment: TextInput.AlignVCenter
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.EditableText
    Accessible.name: label.length > 0 ? label : placeholder
    Accessible.description: hint

    FontMetrics { id: fm; font: control.font }

    onEditingFinished: committed(text)
    onErrorChanged: {
        if (!error)
            return
        if (Theme.reduceMotion) {
            error = false
        } else {
            shakeAnim.restart()
        }
    }

    transform: Translate { id: shift }
    SequentialAnimation {
        id: shakeAnim
        NumberAnimation { target: shift; property: "x"; to: 6; duration: 50; easing.type: Easing.InOutSine }
        NumberAnimation { target: shift; property: "x"; to: -6; duration: 50; easing.type: Easing.InOutSine }
        NumberAnimation { target: shift; property: "x"; to: 6; duration: 50; easing.type: Easing.InOutSine }
        NumberAnimation { target: shift; property: "x"; to: -6; duration: 50; easing.type: Easing.InOutSine }
        NumberAnimation { target: shift; property: "x"; to: 6; duration: 50; easing.type: Easing.InOutSine }
        NumberAnimation { target: shift; property: "x"; to: 0; duration: 50; easing.type: Easing.InOutSine }
        ScriptAction { script: control.error = false }
    }

    background: Item {
        implicitWidth: 120
        implicitHeight: Theme.controlH
        GlassShadow {
            anchors.fill: parent
            radius: control.radius
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: control.radius
            borderColor: control.enabled ? Theme.borderColor : control.palette.disabled.windowText
        }
        FocusRing { ringRadius: control.radius; active: control.activeFocus }
    }

    // Templates do not draw the placeholder themselves.
    Text {
        x: control.leftPadding
        y: control.topPadding
        width: control.width - control.leftPadding - control.rightPadding
        height: control.height - control.topPadding - control.bottomPadding
        visible: control.length === 0 && control.preeditText.length === 0 && control.placeholderText.length > 0
        text: control.placeholderText
        font: control.font
        color: control.placeholderTextColor
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
