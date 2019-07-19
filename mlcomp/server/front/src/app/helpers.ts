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

    public static parse_time(time){
        return new Date(Date.parse(time));
    }

    static clone(item) {
        return JSON.parse(JSON.stringify(item));
    }
}