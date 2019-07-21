import {Component, Input} from "@angular/core";
import {Paginator} from "../paginator";
import {Log, LogFilter} from "../models";
import {Location} from "@angular/common";
import {ActivatedRoute, Router} from "@angular/router";
import {LogService} from "./log.service";


@Component({
    selector: 'app-log',
    templateUrl: './log.component.html',
    styleUrls: ['./log.component.css']
})
export class LogComponent extends Paginator<Log>{
    displayed_columns: string[] = [
        'time',
        'task',
        'component',
        'step',
        'module',
        'line',
        'level',
        'computer',
        'message'];

    dag: number;
    total: number;
    @Input() task: number;
    @Input() step: number;
    @Input() init_level: string;
    private computer: string;

    private debug: boolean = true;
    private info: boolean = true;
    private warning: boolean = true;
    private error: boolean = true;

    private api: boolean = true;
    private supervisor: boolean = true;
    private worker: boolean = true;
    private worker_supervisor: boolean = true;

    private task_name: string = '';
    private step_name: string = '';

    constructor(
        protected service: LogService,
        protected location: Location,
        private router: Router,
        private  route: ActivatedRoute
    ) {
        super(service, location);
    }

    protected _ngOnInit() {
        this.route.queryParams
        .subscribe(params => {
            if(params['dag'])this.dag = parseInt(params['dag']);
            if(params['task'])this.task = parseInt(params['task']);
            if(params['computer'])this.computer = params['computer'];
        });

        if(this.init_level){
            this.debug = this.init_level=='debug';
            this.info = this.init_level=='info';
            this.warning = this.init_level=='warning';
            this.error = this.init_level=='error';
        }
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

        if(this.worker_supervisor){
            components.push(3);
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
        res.step = this.step;

        return res;
    }
}
