export class Helpers {
    public static unique(array, key) {
        let els = [];
        for (let item of array) {
            if (els.indexOf(item[key]) == -1) {
                els.push(item[key]);
            }
        }

        return els;
    }

    public static size(s: number) {
        if(s == 0){
            return '0';
        }
        if (s < Math.pow(2, 10)) {
            return `${s} byte`;
        }
        if (s < Math.pow(2, 20)) {
            return `${Math.floor(s / 1024)} kbyte`;
        }
        if (s < Math.pow(2, 30)) {
            return `${Math.floor(s / Math.pow(2, 20))} mbyte`;
        }

        return `${(s / Math.pow(2, 30)).toFixed(2)} gbyte`;

    }

    public static format_date_time(date, year=false, month_numeric=false) {
        const monthNames = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ];

        var d = new Date(date),
            month = d.getMonth(),
            day = '' + d.getDate();

        if (day.length < 2) day = '0' + day;

        let parts = [];
        if(year){
            parts.push(d.getFullYear());
        }
        if(month_numeric){
            let month_str = String(month);
            if(month_str.length == 1) month_str = '0' + month_str;
            parts.push(month_str);
        }
        else{
            parts.push(monthNames[month]);
        }
        parts.push(day);

        return parts.join('.') + ' ' +
            date.toTimeString().slice(0, 8);
    }

    public static parse_time(time) {
        return new Date(Date.parse(time));
    }

    public static clone(item) {
        return JSON.parse(JSON.stringify(item));
    }

    public static array_to_dict<T>(array: Array<T>, key_column: string) {
        const normalizedObject: any = {};
        for (let i = 0; i < array.length; i++) {
            const key = 'id' + array[i][key_column].toString();
            normalizedObject[key] = array[i]
        }
        return normalizedObject as { [key: string]: T }
    }

    private static update_object_array(d1,
                                       d2,
                                       ids: string[] = []) {
        let id = null;
        for(let e of ids){
            if(e.indexOf('.')==-1){
                id = e;
                break
            }
        }

        if(!id){
            return d2;
        }

        ids.splice(ids.indexOf(id), 1);

        let res_d = Helpers.array_to_dict(d2, id);
        let target_d = Helpers.array_to_dict(d1, id);
        let names = Object.getOwnPropertyNames(res_d);
        for (let k of names) {
            if (k in target_d) {
                Helpers.update_object(target_d[k], res_d[k], ids);

                delete res_d[k];
                delete target_d[k];
            }

        }

        for (let k in target_d) {
            let index = d1.indexOf(target_d[k]);
            d1.splice(index, 1);
        }
        for (let k in res_d) {
            d1.push(res_d[k]);
        }
        d1 = d1.sort(
            (a, b) => a[id] > b[id]
                ? -1 : 1);
        return d1;

    }

    private static update_object_dict(d1, d2, ids: string[]) {
        for (let name in d2) {
            if (!(name in d1)) {
                d1[name] = d2[name];
            }
            if (JSON.stringify(d1[name]) != JSON.stringify(d2[name])) {
                let ids_name = [];
                for (let id of ids) {
                    if (id.startsWith(name + '.')) {
                        let id_name = id.slice(name.length + 1);
                        ids_name.push(id_name);
                    }
                }

                d1[name] = Helpers.update_object(d1[name],
                    d2[name],
                    ids_name);
            }
        }

        for (let name of d1) {
            if (!(name in d2)) {
                delete d1[name];
            }
        }
    }

    public static update_object(d1,
                                d2,
                                ids: string[] = []
    ) {
        if (!d1) {
            return d2;
        }

        let type = typeof (d1);
        if (!d2 || typeof (d2) != type) {
            return d2;
        }
        if (Array.isArray(d1)) {
            return Helpers.update_object_array(d1, d2, ids);
        } else if (type == 'object') {
            Helpers.update_object_dict(d1, d2, ids);
        } else {
            return d2;
        }

        return d1;
    }

    public static handle_textarea_down_key(event, element){
        let content = event.target.value;
        let start = event.target.selectionStart;

        if (event.key == 'Tab' && !event.ctrlKey) {
            event.preventDefault();
            let selection = content.substring(start,
                event.target.selectionEnd);
            let lines = selection.split(/\r?\n/);
            for (let i = 0; i < lines.length; i++) {
                if (!event.shiftKey) {
                    lines[i] = "  " + lines[i];
                } else {
                    for (let j = 0; j < 2; j++) {
                        if (lines[i].length > 0 && lines[i][0] == ' ') {
                            lines[i] = lines[i].slice(1);
                        }
                    }

                }

            }
            selection = lines.reduce((a, c) => a + '\r\n' + c);

            content = content.substring(0, start) +
                selection + content.substring(event.target.selectionEnd);
            if(!event.shiftKey)
            {
                start = start + 2;
            }

            element.value = content;

            element.selectionStart = start;
            element.selectionEnd = start;

            return content;

        }
    }


}