import {
    OnInit,
    ViewChild,
    EventEmitter,
    OnDestroy
} from '@angular/core';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';
import {Location} from '@angular/common';
import {PaginatorFilter, PaginatorRes} from "./models";
import {BaseService} from "./base.service";
import {Helpers} from "./helpers";

export class Paginator<T> implements OnInit, OnDestroy {
    dataSource: MatTableDataSource<T> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();
    data_updated: EventEmitter<any> = new EventEmitter();

    protected displayed_columns: string[] = [];
    protected default_page_size: number = 15;
    isLoading_results = false;
    total: number;
    private interval: number;
    id_column: string = 'id';
    private previous_filter;

    protected constructor(
        protected service: BaseService,
        protected location: Location,
        protected filter_params: CallableFunction = null,
        protected filter_key: string = null,
        protected enable_interval: boolean = true,
        public init: boolean = true
    ) {

    }

    protected _ngOnInit() {

    }

    get_filter(): any {
        let res = new PaginatorFilter();
        res.page_number = this.paginator ? this.paginator.pageIndex : 0;
        res.page_size = this.paginator ?
            this.paginator.pageSize || this.default_page_size : 15;
        if (this.sort) {
            res.sort_column = this.sort.active ? this.sort.active : '';
            res.sort_descending = this.sort.direction ?
                this.sort.direction == 'desc' : true;
        }

        if (this.filter_key) {
            let final = {[this.filter_key]: res};
            if (this.filter_params) {
                final = {...final, ...this.filter_params()};
            }
            return final;
        }

        return res;
    }

    ngOnInit() {
        if (!this.init) {
            return;
        }

        this._ngOnInit();

        // If the user changes the sort order, reset back to the first page.
        if (this.sort) {
            this.sort.sortChange.subscribe(
                () => {
                    if (this.paginator) {
                        this.paginator.pageIndex = 0;
                    }
                }
            );
        }


        let m = merge(this.change);
        if (this.sort) {
            m = merge(m, this.sort.sortChange);
        }
        if (this.paginator) {
            m = merge(m, this.paginator.page);
        }

        m.pipe(
            startWith({}),
            switchMap(() => {
                this.isLoading_results = true;
                let filter = this.get_filter();
                if (!filter) {
                    return observableOf(new PaginatorRes<T>());
                }
                if (this.previous_filter) {
                    let keys = {...Object.keys(this.previous_filter),
                        ...Object.keys(filter)};

                    for (let i in keys) {
                        let k = keys[i];
                        if (JSON.stringify(this.previous_filter[k])
                            != JSON.stringify(filter[k])
                            && k != 'paginator') {
                            this.paginator.pageIndex = 0;
                            if (filter.paginator) {
                                filter.paginator.page_number = 0;
                            } else {
                                filter.page_number = 0;
                            }
                        }
                    }
                }

                this.previous_filter = filter;
                return this.service.get_paginator<T>(filter);
            }),
            map(res => {
                // Flip flag to show that loading has finished.
                this.isLoading_results = false;

                return res;
            }),
            catchError(() => {
                this.isLoading_results = false;
                return observableOf(new PaginatorRes<T>());
            })
        ).subscribe(res => {
            if (!res || !res.data) {
                return;
            }

            this.dataSource.data = Helpers.update_object(
                this.dataSource.data,
                res.data,
                [this.id_column]);

            this.total = res.total;
            this.data_updated.emit(res);
        });

        if (this.enable_interval) {
            this.interval = setInterval(
                () => this.change.emit('event'),
                3000);
        }

    }

    ngOnDestroy() {
        if (this.enable_interval) {
            clearInterval(this.interval);
        }

    }

}
