pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    GlassTextArea - T.TextArea (docs/GLASS_DESIGN.md §2.3): the multi-line
    sibling of GlassTextField for the series prompt template.  Same "fill"
    material and focus ring, word-wrapped, and it GROWS with its content from
    `minLines` rows upward, so the page's own Flickable scrolls the whole
    template - no inner scrolling, no transforms, the clip stays honest.
    `committed(text)` fires on focus loss and on Ctrl+Enter.  `placeholder` is
    drawn by the component (Templates do not).  `label` / `hint` come from
    the FormRow and feed Accessible.*.
*/
T.TextArea {
    id: control

    property string label: ""
    property string hint: ""
    property string placeholder: ""
    property int minLines: 6
    property bool monospace: false
    property real radius: Theme.radiusControl
    signal committed(string text)

    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink
    readonly property real minContentHeight: fm.height * minLines

    implicitWidth: 240
    implicitHeight: Math.round(Math.max(minContentHeight, contentHeight) + topPadding + bottomPadding)
    leftPadding: 12
    rightPadding: 12
    topPadding: Theme.padY + 2
    bottomPadding: Theme.padY + 2
    font.family: monospace ? Theme.mono : Theme.family
    font.pointSize: Theme.fontBody
    color: inkColor
    selectionColor: Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.25)
    selectedTextColor: Theme.ink
    placeholderText: placeholder
    placeholderTextColor: Theme.inkSecondary
    wrapMode: TextEdit.Wrap
    selectByMouse: true
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.EditableText
    Accessible.name: label.length > 0 ? label : placeholder
    Accessible.description: hint
    Accessible.multiLine: true

    FontMetrics { id: fm; font: control.font }

    onEditingFinished: committed(text)
    Keys.onPressed: function (event) {
        if ((event.key === Qt.Key_Return || event.key === Qt.Key_Enter)
                && (event.modifiers & Qt.ControlModifier)) {
            control.committed(control.text)
            event.accepted = true
        }
    }

    background: Item {
        implicitWidth: 240
        implicitHeight: Theme.controlH * 3
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

    Text {
        x: control.leftPadding
        y: control.topPadding
        width: control.width - control.leftPadding - control.rightPadding
        visible: control.length === 0 && control.preeditText.length === 0 && control.placeholderText.length > 0
        text: control.placeholderText
        font: control.font
        color: control.placeholderTextColor
        wrapMode: Text.Wrap
        elide: Text.ElideRight
        maximumLineCount: Math.max(control.minLines, 12)
    }
}
