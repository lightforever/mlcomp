import {Component} from "@angular/core";
import {Paginator} from "../paginator";
import {Log, LogFilter} from "../models";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {MatIconRegistry} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {MessageService} from "../message.service";
import {LogService} from "../log.service";


@Component({
    selector: 'app-log',
    templateUrl: './log.component.html',
    styleUrls: ['./log.component.css']
})
export class LogComponent extends Paginator<Log>{
    displayed_columns: string[] = ['time', 'component', 'level', 'task', 'step', 'computer', 'message'];
    dag: string;
    total: number;
    private task: string;
    private computer: string;

    private debug: boolean = true;
    private info: boolean = true;
    private warning: boolean = true;
    private error: boolean = true;

    private api: boolean = true;
    private supervisor: boolean = true;
    private worker: boolean = true;

    private task_name: string = '';
    private step_name: string = '';

    constructor(
        protected service: LogService,
        protected location: Location,
        private router: Router, private  route: ActivatedRoute,
        iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
        private message_service: MessageService
    ) {
        super(service, location);
    }

    protected _ngOnInit() {
        this.route.queryParams
        .subscribe(params => {
            this.dag = params['dag'];
            this.task = params['task'];
            this.computer = params['computer'];
        });

    }

    apply_task_name(value: string) {
        this.task_name = value;
        this.change.emit();
    }

    apply_step_name(value: string) {
        this.step_name = value;
        this.change.emit();
    }


    get_components() {
        let components: number[] = [];
        if (this.api) {
            components.push(0);
        }
        if (this.supervisor) {
            components.push(1);
        }
        if (this.worker) {
            components.push(2);
        }

        return components;
    }

    get_levels() {
        let levels: number[] = [];
        if (this.debug) {
            levels.push(10);
        }
        if (this.info) {
            levels.push(20);
        }
        if (this.warning) {
            levels.push(30);
        }
        if (this.error) {
            levels.push(40);
        }
        return levels;
    }

    get_filter(): LogFilter{
        let res = new LogFilter();
        res.dag = this.dag;
        res.task = this.task;
        res.task_name  = this.task_name;
        res.components = this.get_components();
        res.computer = this.computer;
        res.levels = this.get_levels();
        res.step_name = this.step_name;
        res.paginator = super.get_filter();

        return res;
    }
}
