pragma ComponentBehavior: Bound
import QtQuick

/*!
    FormRow - one labelled row of a GlassCard: `label` in a fixed column on the
    left, the control(s) on the right, an optional caption `hint` under the
    control.  The content item is stretched to the control column unless
    `stretch` is false.  `label` and `hint` are forwarded to the control's own
    `label` / `hint` properties when they are empty, which the Glass* controls
    bind to Accessible.name / Accessible.description.
*/
Item {
    id: row

    property string label: ""
    property string hint: ""
    property real labelWidth: Math.round(140 * Theme.textScale)
    property bool stretch: true
    default property alias content: slot.data

    width: parent ? parent.width : 400
    implicitWidth: 400
    implicitHeight: Math.max(labelText.implicitHeight, slot.implicitHeight)
                    + (hintText.visible ? hintText.implicitHeight + 2 : 0)

    Text {
        id: labelText
        width: row.labelWidth - Theme.gap
        y: Math.max(0, (slot.implicitHeight - implicitHeight) / 2)
        text: row.label
        wrapMode: Text.WordWrap
        font.family: Theme.family
        font.pointSize: Theme.fontBody
        color: Theme.ink
    }

    Item {
        id: slot
        x: row.labelWidth
        width: row.width - x
        height: implicitHeight
        implicitHeight: {
            var h = 0
            for (var i = 0; i < children.length; ++i) {
                var c = children[i]
                if (c.visible)
                    h = Math.max(h, c.implicitHeight > 0 ? c.implicitHeight : c.height)
            }
            return h
        }
    }

    Text {
        id: hintText
        x: slot.x
        y: slot.height + 2
        width: slot.width
        visible: row.hint.length > 0
        text: row.hint
        wrapMode: Text.WordWrap
        font.family: Theme.family
        font.pointSize: Theme.fontCaption
        color: Theme.inkSecondary
    }

    Component.onCompleted: attach()

    function attach() {
        for (var i = 0; i < slot.children.length; ++i) {
            var c = slot.children[i]
            if (row.stretch)
                c.width = Qt.binding(function () { return slot.width })
            if (c.label !== undefined && c.label === "")
                c.label = Qt.binding(function () { return row.label })
            if (c.hint !== undefined && c.hint === "")
                c.hint = Qt.binding(function () { return row.hint })
        }
    }
}
