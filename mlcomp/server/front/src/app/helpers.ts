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

    public static format_date_time(date) {
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

        return [monthNames[month], day].join('.') + ' ' +
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


}